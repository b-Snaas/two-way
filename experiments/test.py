from former import TransformerBlock, SelfAttention, util

from former.util import d, here, tic, toc
import former

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist
from torch.cuda.amp import autocast, GradScaler

import numpy as np

from argparse import ArgumentParser
import wandb

import random, sys, math, gzip, time

from tqdm import tqdm, trange

import fire

"""
The idea of this experiment is to take the student/teacher distillation idea, and work it into a single model.
- The model always has maximum size, but it produces an output from the last layer and from a layer halfway up.
- After a forward, the output at the top is used as a distillation target for the layer halfway up.    
- This implements the distillation trick, but more efficiently. 

"""

# NB, the enwik8 data contains tokens from 9 to 240, but we'll round up to the nearest
# power of two.
NUM_TOKENS = 256
# Used for converting between nats and bits
LOG2E = math.log2(math.e)

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default', distpoint=None):
        super().__init__()

        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.toprobs = nn.Linear(emb, num_tokens)
        self.toprobsdist = nn.Linear(emb, num_tokens)

        self.distpoint = distpoint

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

        di = None
        for i, block in enumerate(self.tblocks):
            x = block(x) + x
            if i == self.distpoint:
                di = x # this is the output at the distillation point

        x = self.toprobs(x)
        y = None if di is None else self.toprobsdist(di)

        return x, y

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

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

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
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def compute_compression(model, data, context, batch_size):
    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    for current in range(data.size(0)):

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        if instance.size(0) < context + 1:
            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([pad, instance], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

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

            output = F.log_softmax(model(inputs)[0], dim=-1)

            lnprobs = output[torch.arange(b, device=d()), -1, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch = []  # clear the buffer

    return bits / data.size(0) # bits-per-byte

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

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])[0]
        output = F.log_softmax(output, dim=-1)

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed

def copy_params(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)

    for name, param in params_src:
        if name in dict_dest:
            dict_dest[name].data.copy_(param.data)

def validate(model, data, num, context):

    source, target = sample_batch(data, length=context, batch_size=num)

    target = target[:, -1] # we only test the last element

    if torch.cuda.is_available():
        source, target = source.cuda(), target.cuda()

    with torch.no_grad():

        output = model(source)

        lnprobs = output[torch.arange(num, device=d()), -1, target]
        log2probs = lnprobs * LOG2E
        # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
        #    probabilities, since these give us bits.

        return - log2probs.mean()

def go(num_batches=1_000_000, batch_size=32, seed=1, data=None,
       lr=3e-4, tb_dir='./runs', final=False, embedding=768, enrich_loss=False,
       num_heads = 8, context=512, depth=12, test_every=1_000, test_subset=100_000, val_batchsize=128,
       test_batchsize=64, gradient_clipping=1.0, sample_length=600, attention_type='default', distilltemp=2.0, warmup=0,
       distpoint=6, tags=None,
       gamma=None, debug=False, distill_with_target=True, lr_max=1e-3, lr_min=1e-5, peak=0.3, anneal='linear'):

    """

    :param batches:
    :param batch_size:
    :param seed:
    :param data:
    :param lr:
    :param tb_dir:
    :param final:
    :param embedding:
    :param enrich_loss:
    :param num_heads:
    :param context:
    :param depth:
    :param test_every:
    :param test_subset:
    :param val_batchsize:
    :param test_batchsize:
    :param gradient_clipping:
    :param sample_length:
    :param attention_type:
    :param distilltemp:
    :param warmup:
    :param dist_layer:
    :param tags:
    :param gamma: The distillation loss multiplier.
    :return:
    """

    print(locals())

    wd = wandb.init(
        project='self-distillation',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    if seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(seed)

    # load the data (validation unless arg.final is true, then test)
    data = here('data/enwik8.gz') if data is None else data

    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                            if final else (data_train, data_val)

    model = GTransformer(emb=embedding, heads=num_heads, depth=depth, seq_length=context,
                num_tokens=NUM_TOKENS, attention_type=attention_type, distpoint=distpoint)

    # Training loop
    # -- We don't loop over the data, instead we sample a batch of random subsequences each time. This is not strictly
    #    better or worse as a training method, it's just a little simpler.
    instances_seen = 0
    t0 = time.time()

    print(f'Training model.')

    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=lr_max,
        total_steps=num_batches,
        pct_start=peak,
        final_div_factor=(lr_max / lr_min),
        anneal_strategy=anneal
    )
    instances_seen = 0
    scaler = GradScaler()

    if torch.cuda.is_available():
        model.cuda()

    for i in trange(num_batches):

        opt.zero_grad()

        source, target = sample_batch(data_train, length=context, batch_size=batch_size)
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        with autocast():
            output, doutput = model(source)

        # Compute the loss
        loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')

        if gamma is not None:
            # - how closely it mimics the network output
            out = output.transpose(2, 1).detach()
            outp = F.softmax(out, dim=1)
            distill_loss =  F.cross_entropy(doutput.transpose(2, 1), outp, reduction='mean')

            # - how closely it mimics the training target
            if distill_with_target:
                target_loss = F.cross_entropy(doutput.transpose(2, 1),target, reduction='mean')
            else:
                target_loss = 0.0

            total_loss = loss + gamma * (distill_loss + target_loss)

        wandb.log({'gamma': gamma})

        wandb.log({'loss': float(loss.item()) * util.LOG2E})
        if gamma is not None:
            wandb.log({'distill_loss': float(distill_loss.item()) * util.LOG2E})
            wandb.log({'target_loss': float(loss.item()) * util.LOG2E})
            wandb.log({'total_loss': float(total_loss.item()) * util.LOG2E})

        # tbw.add_scalar('distill/train-loss', float(loss.item()) * LOG2E, instances_seen)
        # tbw.add_scalar('distill/time-forward', t, instances_seen)
        # tbw.add_scalar('distill/ts', 0, instances_seen)

        if gamma is not None:
            scaler.scale(total_loss).backward()
        else:
            scaler.scale(loss).backward()

        # Unscale the gradients before clipping
        scaler.unscale_(opt)

        # clip gradients
        # -- If the total gradient vector has a length > x, we clip it back down to x.
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        # Scaler step and update
        scaler.step(opt)
        scaler.update()

        sch.step()

        # lprobs = validate(model=teacher, data=data_val, num=val_batchsize, context=context)
        # tbw.add_scalar('distill/eval', lprobs, instances_seen)
        # tbw.add_scalar('distill/eval-by-time', lprobs, time.time() - t0)

        # Validate every `arg.test_every` steps. First we compute the
        # compression on the validation data (or a subset),
        # then we generate some random text to monitor progress.
        if i % test_every == 0 and i != 0:
            print(f'           validating.')

            with torch.no_grad():

                ## Sample and print a random sequence

                # Slice a random seed from the test data, and sample a continuation from the model.
                seedfr = random.randint(0, data_test.size(0) - context)
                seed = data_test[seedfr:seedfr + context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(model, seed=seed, max_context=context, verbose=True, length=sample_length)

                ## Compute validation bits per byte

                upto = test_subset
                data_sub = data_test[:upto]

                bits_per_byte = compute_compression(model, data_sub, context=context, batch_size=test_batchsize)
                # -- Since we're not computing gradients, we can increase the batch size a little from what we used in
                #    training.

                print(f' model {bits_per_byte:.4} bits per byte')
                wandb.log({'eval': bits_per_byte})
                # tbw.add_scalar(f'distill/student-bpb', bits_per_byte, i * batch_size, instances_seen)
                # tbw.add_scalar(f'distill/student-bpb-time', bits_per_byte, i * batch_size, time.time() - t0)
                #
                # bits_per_byte = compute_compression(teacher, data_sub, context=context, batch_size=test_batchsize)
                # print(f'            teacher {bits_per_byte:.4} bits per byte')
                # tbw.add_scalar(f'distill/teacher-bpb', bits_per_byte, i * batch_size, instances_seen)
                # tbw.add_scalar(f'distill/teacher-bpb-time', bits_per_byte, i * batch_size, time.time() - t0)
                # -- 0.9 bit per byte is around the state of the art.


if __name__ == "__main__":

    fire.Fire(go)