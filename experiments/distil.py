import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from former import util, TransformerBlock
from former.util import here, d
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random, tqdm, gzip, fire, wandb
import pickle

# NB, the enwik8 data contains tokens from 9 to 240, but we'll round up to the nearest power of two.
NUM_TOKENS = 256

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):
        super().__init__()

        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.toprobs = nn.Linear(emb, num_tokens)

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type)
            )

        self.tblocks = nn.ModuleList(modules=tblocks)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        for i, block in enumerate(self.tblocks):
            x = block(x)

        x = self.toprobs(x)

        return x

def load_markov_model(filename):
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    print(f"Markov model loaded from {filename}")
    return models

def get_batch_char_probs(models, contexts, numtokens, smoothing):
    order = 4
    ngrams = [''.join([chr(tok) for tok in context]) for context in contexts]
    model = models[order]

    # Get counts of possible next symbols for each ngram in the batch
    batch_counts = np.array([[model.get(ngram + chr(i), 0) for i in range(numtokens)] for ngram in ngrams])

    # Apply smoothing
    batch_probabilities = (batch_counts + smoothing[order]) / (batch_counts.sum(axis=1, keepdims=True) + smoothing[order] * numtokens)

    return batch_probabilities

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
    # -- the start index is the one we just sampled, and the end is exactly 'length' positions after that.
    seqs_target = [data[start + 1 : start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimension to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
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

# Define a function to get memory usage
def get_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return allocated, reserved

def go(
    num_batches=1_000_000,
    batch_size=32,
    data=None,
    lr_min=1e-4,
    lr_max=3e-4,
    peak=0.2,
    anneal="cos",
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
    pre_trained_model_path=None,
    markov_model_path=None,
    distillation_mode=None,
    gamma=1.0,
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
            "batch_size": batch_size,
            "embedding_size": embedding_size,
            "num_heads": num_heads,
            "context": context,
            "depth": depth,
            "seed": seed,
            "gradient_clipping": gradient_clipping,
        },
    )

    # Load the data (validation unless final is true, then test)
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

    # # Load pre-trained model for distillation
    # if pre_trained_model_path:
    #     state_dict = torch.load(pre_trained_model_path)
    #     # Determine depth by inspecting the keys of the state dictionary
    #     pre_trained_model = GTransformer(
    #         emb=embedding_size,
    #         heads=num_heads,
    #         depth=depth,
    #         seq_length=context,
    #         num_tokens=NUM_TOKENS,
    #         attention_type=attention_type,
    #     )
    #     pre_trained_model.load_state_dict(state_dict)
    #     pre_trained_model.eval()
    #     if torch.cuda.is_available():
    #         pre_trained_model.cuda()

    # Load the 4-gram Markov model
    if distillation_mode == "markov":
        markov_models = load_markov_model(markov_model_path)
        smoothing = [1.0, 0.1, 0.01, 0.0001, 0.00001]

    # Create the model
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
            initial_loss = F.cross_entropy(output.transpose(2, 1), target, reduction="mean")

            if distillation_mode == "pre_trained" and pre_trained_model_path:
                with torch.no_grad():
                    pre_trained_output = pre_trained_model(source)
                out = pre_trained_output.transpose(2, 1).detach()
                outp = F.softmax(out, dim=1)
                distill_loss = F.cross_entropy(output.transpose(2, 1), outp, reduction='mean')
                loss = (1 - gamma) * initial_loss + gamma * distill_loss

            elif distillation_mode == "markov":
                # Extract the last 4 characters for each sequence in the batch
                contexts = source[:, -4:].cpu().numpy()
                # Get the probabilities for all possible next symbols for each context in the batch
                markov_probs = get_batch_char_probs(markov_models, contexts, NUM_TOKENS, smoothing)
                markov_probs = torch.tensor(markov_probs).to(source.device)
                # Cast markov_probs to Long type
                markov_probs = markov_probs.long()
                # Compute the distillation loss in a vectorized way
                distill_loss = F.cross_entropy(output.transpose(2, 1), markov_probs, reduction='mean')
                loss = (1 - gamma) * initial_loss + gamma * distill_loss

            else:
                loss = initial_loss  # Handle the case where no distillation is applied

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

        # Update the learning rate
        sch.step()

        log_data = {}
        log_data["output-layer-loss"] = float(loss.item()) * util.LOG2E

        # Add each additional key directly
        log_data["learning-rate"] = sch.get_last_lr()[0]
        log_data["batches_seen"] = batches_seen

        # Log the data
        wandb.log(log_data, step=instances_seen)

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
                    model, data_sub, context=context, batch_size=test_batchsize, depth=depth
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
