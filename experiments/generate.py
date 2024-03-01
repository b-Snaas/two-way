import gzip
import random
import tqdm

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from former import GTransformer, util
from former.util import d, here, tic, toc

NUM_TOKENS = (
    256  # Rounded up to the nearest power of two from the enwik8 data token range.
)


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution.
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
    Slices out a batch of subsequences from the data.
    """
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)
    seqs_inputs = [data[start : start + length] for start in starts]
    seqs_target = [data[start + 1 : start + length + 1] for start in starts]

    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    return inputs, target


def sample_sequence(
    model,
    seed_sequence,
    context_size,
    sequence_length=600,
    temperature=0.5,
    verbose=False,
):
    """
    Sequentially samples a sequence from the model, token by token.

    Parameters:
    - model: The trained model used for generating the sequence.
    - seed_sequence: The initial sequence of tokens used to start the generation process.
    - context_size: The maximum length of the sequence that the model takes as input.
    - sequence_length: The total length of the sequence to generate, including the seed.
    - temperature: A parameter controlling the randomness of the predictions; lower values make the model more confident, higher values make the predictions more diverse.
    - verbose: If True, prints the generated sequence as it is being generated.

    Returns:
    - The generated sequence as a torch.Tensor.
    """
    print("Starting sequence generation...")

    # Detach the seed sequence from any previous computational graph and clone it to avoid modifying the original.
    generated_sequence = seed_sequence.detach().clone()

    # Print the seed sequence if verbose mode is on.
    if verbose:
        print("Seed sequence: [", end="", flush=True)
        for token in seed_sequence:
            print(str(chr(token)), end="", flush=True)
        print("]", end="", flush=True)

    # Generate new tokens one by one, up to the specified sequence length.
    for _ in range(sequence_length - len(seed_sequence)):
        # Take the last 'context_size' tokens as input for the model.
        input_sequence = generated_sequence[-context_size:]

        # Get the model's prediction for the next token.
        logits = model(input_sequence[None, :])
        next_token = sample(logits[0, -1, :], temperature)

        # Print the generated token if verbose mode is on.
        if verbose:
            print(str(chr(max(32, next_token))), end="", flush=True)

        # Append the generated token to the sequence.
        generated_sequence = torch.cat([generated_sequence, next_token[None]], dim=0)

    print()  # Newline for formatting if verbose mode is on.
    return generated_sequence


def go(arg):
    """
    Main training and generation loop.
    """
    random.seed(arg.seed if arg.seed >= 0 else random.randint(0, 1000000))
    tbw = SummaryWriter(log_dir=arg.tb_dir)  # Tensorboard logging

    # Load the data
    arg.data = here("data/enwik8.gz") if arg.data is None else arg.data
    data_train, data_val, data_test = enwik8(arg.data)
    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if arg.final
        else (data_train, data_val)
    )

    # Initialize the model
    model = GTransformer(
        emb=arg.embedding_size,
        heads=arg.num_heads,
        depth=arg.depth,
        seq_length=arg.context,
        num_tokens=NUM_TOKENS,
        attention_type=arg.attention_type,
    )
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0)
    )

    # Training loop
    instances_seen = 0
    for i in tqdm.trange(arg.num_batches):
        opt.zero_grad()
        source, target = sample_batch(
            data_train, length=arg.context, batch_size=arg.batch_size
        )
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        tic()
        output = model(source)
        t = toc()

        loss = F.nll_loss(output.transpose(2, 1), target, reduction="mean")
        tbw.add_scalar(
            "transformer/train-loss",
            float(loss.item()) * util.LOG2E,
            i * arg.batch_size,
            instances_seen,
        )
        tbw.add_scalar("transformer/time-forward", t, instances_seen)

        loss.backward()
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()
        sch.step()

        # Validation and text generation
        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
            with torch.no_grad():
                # Sample and print a random sequence
                seedfr = random.randint(0, data_test.size(0) - arg.context)
                seed = data_test[seedfr : seedfr + arg.context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(
                    model,
                    seed=seed,
                    max_context=arg.context,
                    verbose=True,
                    length=arg.sample_length,
                )

                # Compute validation bits per byte
                upto = (
                    data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
                )
                data_sub = data_test[:upto]

                bits_per_byte = util.compute_compression(
                    model, data_sub, context=arg.context, batch_size=arg.test_batchsize
                )
                print(f"epoch{i}: {bits_per_byte:.4} bits per byte")
                tbw.add_scalar(
                    f"transformer/eval-loss",
                    bits_per_byte,
                    i * arg.batch_size,
                    instances_seen,
                )


if __name__ == "__main__":
    print("Parsing arguments...")
    parser = ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        "-N",
        "--num-batches",
        dest="num_batches",
        help="Number of batches to train on.",
        default=1000000,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=32,
        type=int,
    )
    parser.add_argument("-D", "--data", dest="data", help="Data file.", default=None)
    parser.add_argument(
        "-l",
        "--learn-rate",
        dest="lr",
        help="Learning rate",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "-T",
        "--tb-dir",
        dest="tb_dir",
        help="Tensorboard logging directory",
        default="./runs",
    )
    parser.add_argument(
        "-f",
        "--final",
        dest="final",
        action="store_true",
        help="Run on the real test set.",
    )
    parser.add_argument(
        "-E",
        "--embedding",
        dest="embedding_size",
        help="Size of the character embeddings.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "-H",
        "--heads",
        dest="num_heads",
        help="Number of attention heads.",
        default=8,
        type=int,
    )
    parser.add_argument(
        "-C",
        "--context",
        dest="context",
        help="Length of the sequences.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        help="Depth of the network (nr. of transformer blocks)",
        default=12,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        dest="seed",
        help="RNG seed. Negative for random",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--test-every",
        dest="test_every",
        help="How many batches between tests.",
        default=1500,
        type=int,
    )
    parser.add_argument(
        "--test-subset",
        dest="test_subset",
        help="A subset for the validation tests.",
        default=100000,
        type=int,
    )
    parser.add_argument(
        "--test-batchsize",
        dest="test_batchsize",
        help="Batch size for validation loss.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--gradient-clipping",
        dest="gradient_clipping",
        help="Gradient clipping.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Learning rate warmup.",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--sample-length",
        dest="sample_length",
        help="Number of character to sample.",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--attention-type",
        dest="attention_type",
        help="Type of self-attention to use.",
        default="default",
        type=str,
    )

    options = parser.parse_args()
    go(options)
