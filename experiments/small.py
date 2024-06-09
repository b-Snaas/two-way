from former import util, GTransformer
from former.util import here
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random, tqdm, gzip, fire, wandb

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
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end="", flush=True)

        sequence = torch.cat(
            [sequence, c[None]], dim=0
        )  # Append the sampled token to the sequence

    print()
    return seed

# Define a function to save the model's state dictionary
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Define a function to get memory usage
def get_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return allocated, reserved

def go(
    num_batches=1_000_000,
    batch_size=32,
    data=None,
    lr=3e-4,
    warmup_steps=10000,
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
    model_save_path="./models/model.pt",
):

    if seed < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
    else:
        torch.manual_seed(seed)

    wandb.init(
        project="distill-transformer",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
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

    # Print GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_mem_free = gpu_mem_total - gpu_mem_reserved - gpu_mem_allocated
        
        print(f"Using GPU: {gpu_name}")
        print(f"Total Memory: {gpu_mem_total:.2f} GB")
        print(f"Reserved Memory: {gpu_mem_reserved:.2f} GB")
        print(f"Allocated Memory: {gpu_mem_allocated:.2f} GB")
        print(f"Free Memory: {gpu_mem_free:.2f} GB")
    else:
        print("No GPU available, using CPU")

    # create the model
    model = GTransformer(
        emb=embedding_size,
        heads=num_heads,
        depth=depth,
        seq_length=context,
        num_tokens=NUM_TOKENS,
        attention_type=attention_type,
    )
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    # Define the warmup scheduler
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup_steps / batch_size), 1.0))

    # Training loop
    instances_seen = 0
    batches_seen = 0
    scaler = GradScaler()

    for i in tqdm.trange(num_batches): 
        batches_seen += 1

        opt.zero_grad()
        source, target = sample_batch(data_train, length=context, batch_size=batch_size)
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        # Wrap the forward pass in an autocast context
        with autocast():
            output = model(source)  # forward pass
            loss = F.nll_loss(output.transpose(2, 1), target, reduction="mean")

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()

        # Unscale the gradients before clipping
        scaler.unscale_(opt)

        # Gradient clipping
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Scaler step and update
        scaler.step(opt)
        scaler.update()

        # Update the learning rate using the warmup scheduler
        sch.step()

        log_data = {}
        log_data["output-layer-loss"] = float(loss.item()) * util.LOG2E

        # Add each additional key directly
        log_data["learning-rate"] = sch.get_last_lr()[0]
        log_data["batches_seen"] = batches_seen

        # Log the data
        wandb.log(log_data, step=instances_seen)

    # Save the model at the end of training
    save_model(model, model_save_path)

if __name__ == "__main__":
    fire.Fire(go)
