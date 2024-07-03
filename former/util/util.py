import torch, os, time, math, tqdm, random, sys, gzip

import torch.nn.functional as F
import torch.distributions as dist
from former import util

import numpy as np


def enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    if path is None:
        path = here("data/enwik8.gz")

    with gzip.open(path) if path.endswith(".gz") else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def enwik8_bytes(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge as a python list of bytes

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """

    if path is None:
        path = here("data/enwik8.gz")

    with gzip.open(path, "r") if path.endswith(".gz") else open(path, "rb") as file:
        all = file.read()

        split = tuple(s / sum(split) for s in split)
        split = tuple(int(s * len(all)) for s in split)

        train, val, test = (
            all[: split[0]],
            all[split[0] : split[0] + split[1]],
            all[split[0] + split[1] :],
        )

        return train, val, test


def enwik8_string(path=None, split=(90, 5, 5)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """

    if path is None:
        path = here("data/enwik8.gz")

    with gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r") as file:
        all = file.read()

        split = tuple(s / sum(split) for s in split)
        split = tuple(int(s * len(all)) for s in split)

        train, val, test = (
            all[: split[0]],
            all[split[0] : split[0] + split[1]],
            all[split[0] + split[1] :],
        )
        return train, val, test


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


def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shifted one position to the right, to provide as a
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


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"


def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", subpath))


def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)


tics = []


def tic():
    tics.append(time.time())


def toc():
    if len(tics) == 0:
        return None
    else:
        return time.time() - tics.pop()


def slice_diag(matrix, l, dv=None):
    """
    Take a batch of attention matrices for relative position encodings and slice out the relevant attentions. These
    are the length l sequences starting at the diagonal

    :param matrix:
    :return:
    """
    if dv is None:
        dv = d(matrix)

    h, w = matrix.size(-2), matrix.size(-1)

    assert w == 2 * l - 1, f"(h, w)= {(h, w)}, l={l}"

    rest = matrix.size()[:-2]

    matrix = matrix.view(-1, h, w)
    b, h, w = matrix.size()

    result = matrix.view(b, -1)
    result = torch.cat([result, torch.zeros(b, l, device=dv)], dim=1)
    assert result.size() == (b, 2 * l * l), f"result.size() {result.size()}"

    result = result.view(b, l, 2 * l)
    result = result[:, :, :l]

    result = result.view(*rest, h, l)
    return result


# Used for converting between nats and bits
LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)


def compute_compression(model, data, context, batch_size, depth, ema_values=None):
    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A single list of integers representing the data
    :param ema_values: Optional list of EMA values for each layer
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    if ema_values:
        # Print EMA values for each layer
        print("EMA values for each layer:")
        for idx, ema in enumerate(ema_values):
            print(f"Layer {idx + 1}: {ema.value}")

        # Select the layer with the lowest EMA value
        best_layer_idx = np.argmin([ema.value for ema in ema_values])
        print(f"Selected layer: {best_layer_idx + 1} (lowest EMA value: {ema_values[best_layer_idx].value})")
    else:
        # Default to the deepest layer if EMA values are not provided
        best_layer_idx = -1
        print("No EMA values provided, using the output of the deepest layer.")

    for current in range(data.size(0)):

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(
            torch.long
        )  # the subsequence of the data to add to the batch
        if instance.size(0) < context + 1:
            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([pad, instance], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert (
                instance.size(0) == context + 1
            )  # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1]  # input
            target = all[:, -1]  # target values

            if ema_values:
                outputs = model(inputs, current_depth=depth)

                # Get the output of the best layer
                output = outputs[best_layer_idx]

            else:
                output = model(inputs)

            # Apply log softmax to the best layer's output to get log probabilities
            log_probs = F.log_softmax(output, dim=-1)

            # Compute log2 probabilities for the best layer
            log2probs = log_probs[torch.arange(b, device=log_probs.device), -1, target] * util.LOG2E

            # Accumulate the bits-per-byte for the selected layer's output
            bits += (-log2probs.sum()).item()
            batch = []  # clear the buffer

    return bits / data.size(0)  # bits-per-byte


def estimate_compression(
    model,
    data,
    nsamples,
    context,
    batch_size,
    verbose=False,
    model_produces_logits=False,
):
    """
    Estimates the compression by sampling random subsequences instead of predicting all characters.

    NB: This doesn't work for GPT-2 style models with super-character tokenization, since the tokens and number of
    characters are mismatched.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.
    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        # current is the character to be predicted

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(
            torch.long
        )  # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        target_indices.append(
            instance.size(0) - 2
        )  # index of the last element of the context

        if instance.size(0) < context + 1:
            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert (
                instance.size(0) == context + 1
            )  # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1]  # input
            target = all[:, -1]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(
                    output.logits, dim=2
                )  # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (
                b,
                context,
            ), f"was: {output.size()}, should be {(b, context, -1)}"

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += (
                -log2probs.sum()
            )  # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples  # total nr of bits used


def distill_loss(output, target, y_outputs, gamma):
    """
    Compute the primary loss with mandatory distillation loss and target loss for multiple layers.
    Distillation loss is computed based on the similarity of each intermediate output to the final
    model output (as teacher output) and direct loss based on the true labels.

    Parameters:
    - output: The model's final output logits.
    - target: The ground truth labels.
    - y_outputs: A list of the model's intermediate output logits at specified layers.
    - gamma: Weight factor for the distillation and target losses.

    Returns:
    - The computed total loss.
    """

    # Teacher loss computation
    teacher_loss = F.cross_entropy(output.transpose(2, 1), target, reduction="mean")

    # Prepare the detached final output for computing distillation losses
    out = output.transpose(2, 1).detach()
    outp = F.softmax(out, dim=1)

    # Initialize student loss
    student_loss = 0

    student_losses = []

    # Compute distillation and ground truth losses for each y_output
    for y in y_outputs:
        # Compute distillation loss for the current intermediate output
        distill_loss = F.cross_entropy(y.transpose(2, 1), outp, reduction="mean")

        # Compute direct ground truth loss for the current intermediate output
        ground_truth_loss = F.cross_entropy(y.transpose(2, 1), target, reduction="mean")

        # Add distillation and ground truth losses to the student loss
        student_loss += (distill_loss * 0.5) + (ground_truth_loss * 0.5)

        student_losses.append(ground_truth_loss)

    # Scale the combined student losses with gamma and add to teacher loss
    loss = teacher_loss + gamma * student_loss

    return loss, teacher_loss, student_losses


def dynamic_distill_loss(target, y_outputs, gamma, ema_values, subseq_length=10, num_subseq=5):
    print(f"Target shape: {target.shape}")
    
    # Find the index of the output with the lowest EMA value to use as the teacher
    teacher_index = ema_values.index(min(ema_values))
    print(f"Teacher index: {teacher_index}")

    # Use the output with the lowest EMA as the teacher
    teacher_output = y_outputs[teacher_index]
    print(f"Teacher output shape: {teacher_output.shape}")

    # Compute the teacher loss first
    teacher_loss = F.cross_entropy(teacher_output.transpose(2, 1), target, reduction="mean")

    # Prepare the detached teacher output for computing distillation losses
    teacher_out = teacher_output.detach()

    # Initialize student loss
    student_loss = 0
    losses = []

    batch_size, vocab_size, target_length = teacher_out.shape
    print(f"Batch size: {batch_size}, Vocab size: {vocab_size}, Target length: {target_length}")

    # Compute distillation and ground truth losses for each y_output
    for idx, y in enumerate(y_outputs):

        if idx == teacher_index:
            # Append the previously computed teacher ground truth loss
            losses.append(teacher_loss)
        else:
            # Compute direct ground truth loss for the current intermediate output
            ground_truth_loss = F.cross_entropy(y.transpose(2, 1), target, reduction="mean")
            losses.append(ground_truth_loss)

            # Compute distillation loss on subsequences
            # Randomly choose start indices for subsequences
            starts = torch.randint(0, target_length - subseq_length + 1, (num_subseq,))
            print(f"Starts shape: {starts.shape}")
            print(f"Starts: {starts}")

            # Create index tensor for gathering subsequences
            indices = starts[:, None, None] + torch.arange(subseq_length)[None, :, None]
            indices = indices.expand(-1, -1, batch_size).permute(2, 1, 0)
            print(f"Indices shape: {indices.shape}")
            print(f"Indices: {indices}")

            # Extract subsequences for teacher and student
            teacher_seqs = torch.gather(teacher_out.permute(0, 2, 1), 1, indices.to(teacher_out.device))
            student_seqs = torch.gather(y.permute(0, 2, 1), 1, indices.to(y.device))
            print(f"Teacher seqs shape: {teacher_seqs.shape}")
            print(f"Teacher seqs:\n{teacher_seqs}")
            print(f"Student seqs shape: {student_seqs.shape}")
            print(f"Student seqs:\n{student_seqs}")

            # Compute log probabilities
            teacher_log_probs = F.log_softmax(teacher_seqs, dim=2)
            student_log_probs = F.log_softmax(student_seqs, dim=2)
            print(f"Teacher log probs shape: {teacher_log_probs.shape}")
            print(f"Teacher log probs:\n{teacher_log_probs}")
            print(f"Student log probs shape: {student_log_probs.shape}")
            print(f"Student log probs:\n{student_log_probs}")

            # Compute sequence probabilities
            teacher_seq_probs = torch.exp(teacher_log_probs.sum(dim=1))
            student_seq_log_probs = student_log_probs.sum(dim=1)
            print(f"Teacher seq probs shape: {teacher_seq_probs.shape}")
            print(f"Teacher seq probs:\n{teacher_seq_probs}")
            print(f"Student seq log probs shape: {student_seq_log_probs.shape}")
            print(f"Student seq log probs:\n{student_seq_log_probs}")

            # Compute KL divergence for all subsequences
            kl_div = -(teacher_seq_probs * student_seq_log_probs).sum(dim=1).mean()
            print(f"KL divergence: {kl_div.item()}")

            student_loss += kl_div

    # Scale the combined student losses with gamma and add to deepest loss
    loss = losses[-1] + gamma * student_loss

    return loss, teacher_loss, losses, student_loss


def progressive_distill_loss(target, outputs, train_stage, gamma):
    """
    Compute the loss based on the current training stage.
    
    Parameters:
    - target: The ground truth labels.
    - outputs: A list of the model's intermediate output logits at specified layers.
    - train_stage: The current training stage (initial, train, distill).
    - gamma: Weight factor for the distillation loss.
    
    Returns:
    - The computed loss for backpropagation.
    - A list of ground truth losses for logging.
    """
    losses = []
    student_loss = 0

    for idx, output in enumerate(outputs):
        if output is None:
            continue
        
        transposed_output = output.transpose(2, 1)
        ground_truth_loss = F.cross_entropy(transposed_output, target, reduction="mean")
        losses.append(ground_truth_loss)

        if train_stage == "distill" and idx == len(outputs) - 1 and len(outputs) > 1:
            teacher_output = outputs[idx - 1]
            teacher_out = teacher_output.transpose(2, 1).detach()
            teacher_probs = F.softmax(teacher_out, dim=1)
            distill_loss = F.cross_entropy(transposed_output, teacher_probs, reduction="mean")
            combined_loss = (1 - gamma) * ground_truth_loss + gamma * distill_loss
            student_loss = combined_loss

    if train_stage == "distill" and len(outputs) > 1:
        loss = student_loss
    else:
        loss = losses[-1]

    return loss, losses


def ema_update(old, new, beta=0.50):
    """ Update the exponential moving average (EMA) with a new data point. """
    return beta * old + (1 - beta) * new

def compute_ema_losses(outputs, target, ema_losses):
    """
    Compute the cross-entropy loss for each output, update the EMA of the losses,
    and return both the computed losses and the updated EMA losses.
    
    Args:
        outputs (list of torch.Tensor): A list of tensors, where each tensor
            is the logits output from a different layer or distillation point.
        target (torch.Tensor): The ground truth labels that correspond to the main task.
        ema_losses (list of float): Current EMA loss values for each distillation point.

    Returns:
        tuple: A tuple containing:
            - list of torch.Tensor: The computed cross-entropy losses for each output.
            - list of float: Updated EMA loss values.
    """
    losses = []
    for i, output in enumerate(outputs):
        loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')
        losses.append(loss)
        
        if ema_losses[i] == float('inf'):
            ema_losses[i] = loss.item()
        else:
            ema_losses[i] = ema_update(ema_losses[i], loss.item())

    return losses, ema_losses