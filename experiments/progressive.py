from former import util, TwowayGen
from former.util import here, dynamic_distill_loss
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


# Define a function to update the learning rate based on the depth
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

def update_optimizer_lr(optimizer, current_depth, batches_seen, batch_size, lr_by_depth, warmup_steps):
    lr = lr_by_depth[current_depth]
    if batches_seen < warmup_steps:
        lr *= batches_seen / warmup_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def prepare_batch(data_train, context, batch_size, instances_seen):
    source, target = sample_batch(data_train, length=context, batch_size=batch_size)
    instances_seen += source.size(0)
    if torch.cuda.is_available():
        source, target = source.cuda(), target.cuda()
    return source, target

def calculate_loss(model, source, target, current_depth, gamma, ema_values):
    outputs = model(source, current_depth=current_depth)
    valid_outputs = [output for output in outputs if output is not None]
    current_ema_values = [ema.value for ema in ema_values[:len(valid_outputs)]]

    if len(valid_outputs) > 1:
        loss, teacher_loss, ground_truth_losses = dynamic_distill_loss(
            target, valid_outputs, gamma=gamma, ema_values=current_ema_values
        )
    else:
        loss = F.cross_entropy(valid_outputs[0].transpose(2, 1), target, reduction="mean")
        teacher_loss = loss
        ground_truth_losses = [loss]
    
    return loss, teacher_loss, ground_truth_losses

def update_ema_values(ema_values, ground_truth_losses, current_depth_index):
    if current_depth_index < len(ema_values) and current_depth_index < len(ground_truth_losses):
        ema_values[current_depth_index].update(ground_truth_losses[current_depth_index])


def log_training_data(wandb, opt, batches_seen, instances_seen, current_depth, ground_truth_losses, grad_norm, ema_values):
    # Log the learning rate, the current depth, the loss, and the gradient norm
    log_data = {
        "learning-rate": opt.param_groups[0]['lr'],
        "batches_seen": batches_seen,
        "current_depth": current_depth,
        "output-layer-loss": ground_truth_losses[-1].item() * util.LOG2E,
        "gradient-norm": grad_norm.item()
    }

    for idx, ema in enumerate(ema_values):
        log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

    wandb.log(log_data, step=instances_seen)

def evaluate_model(wandb, model, data_test, context, test_subset, test_batchsize, batches_seen, final_batches, test_every, depth, ema_values, instances_seen):
    if batches_seen % test_every == 0:
        with torch.no_grad():
            seedfr = random.randint(0, data_test.size(0) - context)
            seed = data_test[seedfr: seedfr + context].to(torch.long)

            if torch.cuda.is_available():
                seed = seed.cuda()

            upto = test_subset
            data_sub = data_test[:upto]
            bits_per_byte = util.compute_compression(
                model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
            )

            print(f"epoch{batches_seen}: {bits_per_byte:.4} bits per byte")
            wandb.log({"transformer/validation-bits-per-byte": bits_per_byte}, step=instances_seen)

def go(
    progressive_batches=10000,
    final_batches=100000,
    test_every=1500,
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
    test_subset=100000,
    nsamples=64,
    test_batchsize=64,
    gradient_clipping=1.0,
    sample_length=200,
    attention_type="default",
    warmup_steps=10000
):

    if seed < 0:
        seed = random.randint(0, 1000000)
        print("random seed: ", seed)
    else:
        torch.manual_seed(seed)

    quarter_depth = depth // 4

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
        project="progressive-transformer",
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
            "batch_size_by_depth": batch_size_by_depth
        },
    )

    data = here("data/enwik8.gz") if data is None else data

    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if final
        else (data_train, data_val)
    )

    model = TwowayGen(
        emb=embedding_size,
        heads=num_heads,
        depth=depth,
        seq_length=context,
        num_tokens=NUM_TOKENS,
        attention_type=attention_type,
        sep_layers=sep_layers,
    )
    if torch.cuda.is_available():
        model.cuda()

    instances_seen = 0
    batches_seen = 0
    scaler = GradScaler()
    opt = torch.optim.Adam(lr=min(lr_by_depth.values()), params=model.parameters())

    ema1 = ExponentialMovingAverage(decay=0.50)
    ema1.update(1000)
    ema2 = ExponentialMovingAverage(decay=0.50)
    ema2.update(1000)
    ema3 = ExponentialMovingAverage(decay=0.50)
    ema3.update(1000)
    ema4 = ExponentialMovingAverage(decay=0.50)
    ema4.update(1000)

    ema_values = [ema1, ema2, ema3, ema4]

    depths_sequence = [quarter_depth, 2 * quarter_depth, 3 * quarter_depth, depth]

    for depth_idx, current_depth in enumerate(depths_sequence):
        print(f"Training with current depth: {current_depth}")

        for _ in tqdm.trange(progressive_batches):
            batches_seen += 1
            batch_size = batch_size_by_depth[current_depth]

            update_optimizer_lr(opt, current_depth, batches_seen, batch_size, lr_by_depth, warmup_steps)
            source, target = prepare_batch(data_train, context, batch_size, instances_seen)

            opt.zero_grad()
            with autocast():
                loss, teacher_loss, ground_truth_losses = calculate_loss(model, source, target, current_depth, gamma, ema_values)

            update_ema_values(ema_values, ground_truth_losses, depth_idx)

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

            for idx, ema in enumerate(ema_values):
                log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

            wandb.log(log_data, step=instances_seen)

            if batches_seen % test_every == 0:
                with torch.no_grad():
                    seedfr = random.randint(0, data_test.size(0) - context)
                    seed = data_test[seedfr: seedfr + context].to(torch.long)

                    if torch.cuda.is_available():
                        seed = seed.cuda()

                    upto = test_subset
                    data_sub = data_test[:upto]
                    bits_per_byte = util.compute_compression(
                        model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
                    )

                    print(f"epoch{batches_seen}: {bits_per_byte:.4} bits per byte")
                    wandb.log({"transformer/validation-bits-per-byte": bits_per_byte}, step=instances_seen)


        if depth_idx < len(depths_sequence) - 1:
            next_depth = depths_sequence[depth_idx + 1]
            next_depth_idx = depth_idx + 1
            print(f"Distilling to next depth: {next_depth}")

            while True:
                batches_seen += 1
                batch_size = batch_size_by_depth[next_depth]

                update_optimizer_lr(opt, next_depth, batches_seen, batch_size, lr_by_depth, warmup_steps)
                source, target = prepare_batch(data_train, context, batch_size, instances_seen)

                opt.zero_grad()
                with autocast():
                    loss, teacher_loss, ground_truth_losses = calculate_loss(model, source, target, next_depth, gamma, ema_values)

                update_ema_values(ema_values, ground_truth_losses, next_depth_idx)

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

                for idx, ema in enumerate(ema_values):
                    log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

                wandb.log(log_data, step=instances_seen)

                if batches_seen % test_every == 0:
                    with torch.no_grad():
                        seedfr = random.randint(0, data_test.size(0) - context)
                        seed = data_test[seedfr: seedfr + context].to(torch.long)

                        if torch.cuda.is_available():
                            seed = seed.cuda()

                        upto = test_subset
                        data_sub = data_test[:upto]
                        bits_per_byte = util.compute_compression(
                            model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
                        )

                        print(f"epoch{batches_seen}: {bits_per_byte:.4} bits per byte")
                        wandb.log({"transformer/validation-bits-per-byte": bits_per_byte}, step=instances_seen)


                current_ema = ema_values[next_depth // quarter_depth - 1].value
                previous_ema = ema_values[current_depth // quarter_depth - 1].value

                # Log the EMA values to debug the loop condition
                print(f"Checking loop exit condition: current EMA: {current_ema}, previous EMA: {previous_ema}")

                if current_ema < previous_ema:
                    print("Exiting depth transition loop.")
                    break
                else:
                    print("Continuing depth transition loop.")


            print(f"Training with next depth without distillation: {next_depth}")
            for _ in tqdm.trange(progressive_batches):
                batches_seen += 1
                batch_size = batch_size_by_depth[next_depth]

                update_optimizer_lr(opt, next_depth, batches_seen, batch_size, lr_by_depth, warmup_steps)
                source, target = prepare_batch(data_train, context, batch_size, instances_seen)

                opt.zero_grad()
                with autocast():
                    loss, teacher_loss, ground_truth_losses = calculate_loss(model, source, target, next_depth, gamma, ema_values)

                update_ema_values(ema_values, ground_truth_losses, next_depth_idx)

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

                for idx, ema in enumerate(ema_values):
                    log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

                wandb.log(log_data, step=instances_seen)

                if batches_seen % test_every == 0:
                    with torch.no_grad():
                        seedfr = random.randint(0, data_test.size(0) - context)
                        seed = data_test[seedfr: seedfr + context].to(torch.long)

                        if torch.cuda.is_available():
                            seed = seed.cuda()

                        upto = test_subset
                        data_sub = data_test[:upto]
                        bits_per_byte = util.compute_compression(
                            model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
                        )

                        print(f"epoch{batches_seen}: {bits_per_byte:.4} bits per byte")
                        wandb.log({"transformer/validation-bits-per-byte": bits_per_byte}, step=instances_seen)


    print(f"Final training for final depth: {depth}")
    for _ in tqdm.trange(final_batches):
        batches_seen += 1
        batch_size = batch_size_by_depth[depth]

        update_optimizer_lr(opt, depth, batches_seen, batch_size, lr_by_depth, warmup_steps)
        source, target = prepare_batch(data_train, context, batch_size, instances_seen)

        opt.zero_grad()
        with autocast():
            loss, teacher_loss, ground_truth_losses = calculate_loss(model, source, target, depth, gamma, ema_values)

        update_ema_values(ema_values, ground_truth_losses, depth_idx)

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

        for idx, ema in enumerate(ema_values):
            log_data[f"ema-{idx}"] = ema.value if isinstance(ema.value, (int, float)) else ema.value.item()

        wandb.log(log_data, step=instances_seen)

        if batches_seen % test_every == 0:
            with torch.no_grad():
                seedfr = random.randint(0, data_test.size(0) - context)
                seed = data_test[seedfr: seedfr + context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                upto = test_subset
                data_sub = data_test[:upto]
                bits_per_byte = util.compute_compression(
                    model, data_sub, context=context, batch_size=test_batchsize, depth=depth, ema_values=ema_values
                )

                print(f"epoch{batches_seen}: {bits_per_byte:.4} bits per byte")
                wandb.log({"transformer/validation-bits-per-byte": bits_per_byte}, step=instances_seen)


if __name__ == "__main__":
    fire.Fire(go)
