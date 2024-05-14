from former import util, TwowayGen
from former.util import here, dynamic_distill_loss
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler
import numpy as np
import random, tqdm, gzip, fire, wandb
import gc

import warnings

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.
    """
    print("Loading enwik8 dataset...")
    with gzip.open(path) if path.endswith(".gz") else open(path, "rb") as file:
        data = file.read(n_train + n_valid + n_test)
        X = np.frombuffer(data, dtype=np.uint8).copy()
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs = [data[start : start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1 : start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target


def sample_sequence(
    model, seed, max_context, length=600, temperature=0.5, verbose=False
):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose:  # Print the seed, surrounded by square brackets
        print("[", end="", flush=True)
        for c in seed:
            print(str(chr(c)), end="", flush=True)
        print("]", end="", flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :], current_depth=12)[-1]

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end="", flush=True)

        sequence = torch.cat(
            [sequence, c[None]], dim=0
        )  # Append the sampled token to the sequence

    print()
    return seed

class ExponentialMovingAverage:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

# Define a function to get memory usage
def get_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return allocated, reserved

def go(
    num_batches=1_000_000,
    data=None,
    lr_min=1e-4,
    lr_max=3e-4,
    peak=0.2,
    anneal= "cos",
    tb_dir="./runs",
    final=False,
    embedding_size=768,
    num_heads=8,
    context=128,
    depth=12,
    seed=1,
    test_every=1500,
    test_subset=100000,
    nsamples=64,
    test_batchsize=64,
    gradient_clipping=1.0,
    sample_length=200,
    attention_type="default",
):

    if seed < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
    else:
        torch.manual_seed(seed)

    wandb.init(
        project="distill-transformer",
        config={
            "min_learning_rate": lr_min,
            "max_learning_rate": lr_max,
            "embedding_size": embedding_size,
            "num_heads": num_heads,
            "context": context,
            "depth": depth,
            "seed": seed,
            "gradient_clipping": gradient_clipping,
        },
    )

    # load the data (validation unless final is true, then test)
    data = here("data/enwik8.gz") if data is None else data

    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if final
        else (data_train, data_val)
    )

    # create the model
    model = TwowayGen(
        emb=embedding_size,
        heads=num_heads,
        depth=depth,
        seq_length=context,
        num_tokens=NUM_TOKENS,
        attention_type=attention_type,
    )
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=lr_min, params=model.parameters())

    sch = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=lr_max,
        total_steps=num_batches,
        pct_start=peak,
        final_div_factor=(lr_max / lr_min),
        anneal_strategy=anneal
    )

    # Training loop
    instances_seen = 0
    batches_seen = 0
    scaler = GradScaler()

    batch_size_by_depth = {
        1: 500,
        2: 400,
        3: 350,
        4: 300,
        5: 260,
        6: 220,
        7: 190,
        8: 165,
        9: 150,
        10: 140,
        11: 130,
        12: 125
    }

    ema_values = [ExponentialMovingAverage(decay=0.50) for _ in range(depth)]
    for ema in ema_values:
        ema.update(1000)

    for i in tqdm.trange(num_batches):
        batches_seen += 1
        # Randomly choose the current depth for this batch from predefined options
        current_depth = random.choice([d for d in batch_size_by_depth.keys()])
        batch_size = batch_size_by_depth[current_depth]

        # Prepare the batch
        source, target = sample_batch(data_train, length=context, batch_size=135)
        instances_seen += source.size(0)

        # Move data to GPU if available
        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        # Get memory usage before model computation
        allocated_before, reserved_before = get_memory_usage()

        # Gradient zeroing and autocasting
        opt.zero_grad()
        with autocast():
            # Get all layer outputs up to the current maximum depth
            outputs = model(source, current_depth=current_depth)

            # Exclude None values from outputs and prepare for distillation
            valid_outputs = [output for output in outputs if output is not None]

            current_ema_values = [ema.value for ema in ema_values[:len(valid_outputs)]]

            if len(valid_outputs) > 1:
                loss, teacher_loss, ground_truth_losses = dynamic_distill_loss(target, valid_outputs, gamma=0.5, ema_values=current_ema_values)
            else:
                loss = F.cross_entropy(valid_outputs[0].transpose(2, 1), target, reduction="mean")
                teacher_loss = loss
                ground_truth_losses = [loss]

        # Get memory usage after model computation
        allocated_after, reserved_after = get_memory_usage()

        for idx, ema in enumerate(ema_values):
            if idx < len(ground_truth_losses):
                ema.update(ground_truth_losses[idx])

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        # Gradient clipping
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Optimizer and scaler steps
        scaler.step(opt)
        scaler.update()

        # Scheduler step
        sch.step()

        # Update EMAs and log data
        log_data = {
            "learning-rate": sch.get_last_lr()[0],
            "batches_seen": batches_seen,
            "output-layer-loss": ground_truth_losses[-1].item() * util.LOG2E
        }

        for idx, loss in enumerate(ground_truth_losses):
            log_data[f"train-loss-{idx}"] = loss.item() * util.LOG2E

        # Log the data to wandb
        wandb.log(log_data, step=instances_seen)

        # Print the current depth
        print(f"Current depth: {current_depth}")
        print(f"Memory Allocated Before: {allocated_before / (1024 ** 3):.2f} GB, After: {allocated_after / (1024 ** 3):.2f} GB")
        print(f"Memory Reserved Before: {reserved_before / (1024 ** 3):.2f} GB, After: {reserved_after / (1024 ** 3):.2f} GB")

        # Validate every `test_every` steps. First we compute the
        # compression on the validation data (or a subset),
        # then we generate some random text to monitor progress.
        if i != 0 and (i % test_every == 0 or i == num_batches - 1):
            with torch.no_grad():

                ## Sample and print a random sequence

                # Slice a random seed from the test data, and sample a continuation from the model.
                seedfr = random.randint(0, data_test.size(0) - context)
                seed = data_test[seedfr : seedfr + context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(
                    model,
                    seed=seed,
                    max_context=context,
                    verbose=True,
                    length=sample_length,
                )

                ## Compute validation bits per byte

                upto = data_test.size(0) if i == num_batches - 1 else test_subset
                data_sub = data_test[:upto]
                bits_per_byte = util.compute_compression(
                    model, data_sub, context=context, batch_size=test_batchsize
                )
                # -- Since we're not computing gradients, we can increase the batch size a little from what we used in
                #    training.

                print(f"epoch{i}: {bits_per_byte:.4} bits per byte")
                wandb.log(
                    {"transformer/validation-bits-per-byte": bits_per_byte},
                    step=instances_seen,
                )

                # -- 0.9 bit per byte is around the state of the art.


if __name__ == "__main__":
    fire.Fire(go)
