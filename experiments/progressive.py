from former import util, TwowayGen
from former.util import here, progressive_distill_loss
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random, tqdm, gzip, fire, wandb
import copy

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
        output = model(input[None, :], current_depth=12)

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
        if isinstance(new_value, torch.Tensor):
            new_value = new_value.cpu().item()
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

def update_lr(opt, current_depth, step, batch_size, lr_by_depth, warmup_steps):
    for param_group in opt.param_groups:
        base_lr = lr_by_depth[current_depth]
        warmup_factor = min(step / (warmup_steps / batch_size), 1.0)
        param_group['lr'] = base_lr * warmup_factor

# Define a function to get memory usage
def get_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return allocated, reserved

def update_ema_values(ema_values, ground_truth_losses, current_depth_index):
    if current_depth_index < len(ema_values) and current_depth_index < len(ground_truth_losses):
        ema_values[current_depth_index].update(ground_truth_losses[current_depth_index])

def freeze_layers(layers, layer_names):
    print(f"Freezing layers: {layer_names}")
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

def unfreeze_layers(layers, layer_names):
    print(f"Unfreezing layers: {layer_names}")
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = True

def get_layer_params(layers):
    return [copy.deepcopy(layer.state_dict()) for layer in layers]

def compare_layer_params(before, after, layer_names):
    for b, a, name in zip(before, after, layer_names):
        for key in b.keys():
            if not torch.equal(b[key], a[key]):
                print(f"Layer {name} parameter {key} has changed.")

def go(
    data=None,
    tb_dir="./runs",
    final=False,
    embedding_size=768,
    num_heads=8,
    context=128,
    depth=12,
    gamma=0.5,
    decay=0.5,
    sep_layers=False,
    seed=1,
    test_every=1500,
    test_subset=100000,
    nsamples=64,
    test_batchsize=64,
    gradient_clipping=1.0,
    sample_length=200,
    attention_type="default",
    intermediate_amount=10000,
    final_amount=50000,
    warmup_steps=5000
):

    if seed < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
    else:
        torch.manual_seed(seed)

    quarter_depth = depth // 4

    if depth == 12:
        batch_size_by_depth = {
            quarter_depth: 465,
            2 * quarter_depth: 255,
            3 * quarter_depth: 175,
            depth: 130
        }
        lr_by_depth = {
            quarter_depth: 1e-3,
            2 * quarter_depth: 5e-4,
            3 * quarter_depth: 3e-4,
            depth: 1e-4
        }
    elif depth == 24:
        batch_size_by_depth = {
            quarter_depth: 230,
            2 * quarter_depth: 130,
            3 * quarter_depth: 85,
            depth: 65
        }
        lr_by_depth = {
            quarter_depth: 5e-4,
            2 * quarter_depth: 3e-4,
            3 * quarter_depth: 1e-4,
            depth: 5e-5
        }

    wandb.init(
        project="distill-transformer",
        config={
            "embedding_size": embedding_size,
            "num_heads": num_heads,
            "context": context,
            "depth": depth,
            "gamma": gamma,
            "decay": decay,
            "sep_layers": sep_layers,
            "seed": seed,
            "gradient_clipping": gradient_clipping,
            "lr_by_depth": lr_by_depth,
            "batch_size_by_depth": batch_size_by_depth,
            "warmup_steps": warmup_steps
        },
    )

    # Load the data
    data = here("data/enwik8.gz") if data is None else data
    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) if final else (data_train, data_val)

    # Create the model
    model = TwowayGen(
        emb=embedding_size,
        heads=num_heads,
        depth=depth,
        seq_length=context,
        num_tokens=NUM_TOKENS,
        attention_type=attention_type,
        sep_layers=False
    )
    if torch.cuda.is_available():
        model.cuda()

    # Training loop
    instances_seen = 0
    batches_seen = 0
    batches_seen_per_layer = 0
    distillation_batches = 0
    scaler = GradScaler()
    opt = torch.optim.Adam(lr=min(lr_by_depth.values()), params=model.parameters())

    ema_values = [ExponentialMovingAverage(decay=decay) for _ in range(depth // quarter_depth)]
    for ema in ema_values:
        ema.update(1000)

    current_depth = quarter_depth
    depth_index = 0
    train_stage = "initial"
    total_intermediate_batches = sum(intermediate_amount * (i + 1) for i in range((depth // quarter_depth) - 1))
    total_batches = total_intermediate_batches + final_amount
    layer_batches = [intermediate_amount * (i + 1) for i in range((depth // quarter_depth) - 1)]

    prev_params = None

    while batches_seen < total_batches:
        batches_seen += 1
        batches_seen_per_layer += 1
        if train_stage == "distill":
            distillation_batches += 1
            # Freeze all layers up to and including the current distillation layer if gamma is not 0
            if gamma != 0:
                tblock_layers_to_freeze = model.tblocks[:current_depth]
                dist_layers_to_freeze = [model.dist_layers[depth_index]]
                freeze_layers(tblock_layers_to_freeze, [f'tblock_{i}' for i in range(current_depth)])
                freeze_layers(dist_layers_to_freeze, [f'dist_layer_{depth_index}'])

                if prev_params is None:
                    prev_params = get_layer_params(tblock_layers_to_freeze + dist_layers_to_freeze)
        else:
            # Unfreeze all layers if gamma is not 0
            if gamma != 0:
                tblock_layers_to_unfreeze = model.tblocks[:current_depth]
                dist_layers_to_unfreeze = [model.dist_layers[depth_index]]
                unfreeze_layers(tblock_layers_to_unfreeze, [f'tblock_{i}' for i in range(current_depth)])
                unfreeze_layers(dist_layers_to_unfreeze, [f'dist_layer_{depth_index}'])
                
                if prev_params is not None:
                    current_params = get_layer_params(tblock_layers_to_unfreeze + dist_layers_to_unfreeze)
                    compare_layer_params(prev_params, current_params, [f'tblock_{i}' for i in range(current_depth)] + [f'dist_layer_{depth_index}'])
                    prev_params = None

        batch_size = batch_size_by_depth[current_depth]

        update_lr(opt, current_depth, batches_seen_per_layer, batch_size, lr_by_depth, warmup_steps)

        source, target = sample_batch(data_train, length=context, batch_size=batch_size)
        instances_seen += source.size(0)
        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        opt.zero_grad()

        with autocast():
            outputs = model(source, current_depth=current_depth)
            valid_outputs = [output for output in outputs if output is not None]

            loss, ground_truth_losses = progressive_distill_loss(
                target, valid_outputs, train_stage, gamma
            )

        update_ema_values(ema_values, ground_truth_losses, depth_index)
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        scaler.step(opt)
        scaler.update()

        log_data = {
            "learning-rate": opt.param_groups[0]['lr'],
            "batches_seen": batches_seen,
            "current_depth": current_depth,
            "output-layer-loss": ground_truth_losses[-1].item() * util.LOG2E,
            "gradient-norm": grad_norm.item()
        }

        for idx, loss in enumerate(ground_truth_losses):
            log_data[f"train-loss-{idx}"] = loss.item() * util.LOG2E

        for idx, ema in enumerate(ema_values):
            log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

        wandb.log(log_data, step=instances_seen)

        if train_stage != "distill" and (batches_seen - distillation_batches) % test_every == 0 or batches_seen == total_batches:
            with torch.no_grad():
                seedfr = random.randint(0, data_test.size(0) - context)
                seed = data_test[seedfr: seedfr + context].to(torch.long)
                if torch.cuda.is_available():
                    seed = seed.cuda()

                upto = data_test.size(0) if batches_seen == total_batches else test_subset
                data_sub = data_test[:upto]
                bits_per_byte = util.compute_compression(
                    model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
                )

                print(f"Batch {batches_seen}: {bits_per_byte:.4} bits per byte")
                wandb.log(
                    {"transformer/validation-bits-per-byte": bits_per_byte},
                    step=instances_seen,
                )

        # Logic for adding layers and training stages
        if current_depth < depth:
            if train_stage == "initial":
                if batches_seen_per_layer >= layer_batches[depth_index]:
                    depth_index += 1
                    current_depth = (depth_index + 1) * quarter_depth
                    train_stage = "distill"
                    batches_seen_per_layer = 0
                    print(f"Adding layer: {current_depth}")
            elif train_stage == "distill":
                if ema_values[depth_index].value < ema_values[depth_index - 1].value:
                    train_stage = "train"
                    intermediate_batches_seen = 0
                    print(f"Layer {current_depth} EMA has crossed previous layer EMA")
            elif train_stage == "train":
                intermediate_batches_seen += 1
                if intermediate_batches_seen >= layer_batches[depth_index]:
                    if current_depth == depth:
                        train_stage = "final"
                    else:
                        depth_index += 1
                        current_depth = (depth_index + 1) * quarter_depth
                        train_stage = "distill"
                        batches_seen_per_layer = 0
                    print(f"Adding layer: {current_depth}")
        elif train_stage == "final":
            if batches_seen_per_layer >= final_amount:
                print(f"Final training phase completed at batch {batches_seen}")
                break

if __name__ == "__main__":
    fire.Fire(go)