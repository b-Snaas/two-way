import torch
import numpy as np
import gzip
from collections import Counter
import tqdm
import time
import pickle

# Load the enwik8 dataset from the Hutter challenge.
def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    print("Loading enwik8 dataset...")
    with gzip.open(path) if path.endswith(".gz") else open(path, "rb") as file:
        data = file.read(n_train + n_valid + n_test)
        X = np.frombuffer(data, dtype=np.uint8).copy()
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

# Function to measure the time taken
def tic():
    global _start_time
    _start_time = time.time()

def toc():
    if '_start_time' in globals():
        return time.time() - _start_time
    return None

# Function to compute the compression length
def codelength(models, data, len_train, numtokens, smoothing, verbose=False):
    ran = tqdm.trange if verbose else range

    res = [0.0] * len(models)

    if type(smoothing) is float:
        smoothing = [smoothing] * len(models)

    for i in ran(len(data)):
        lprob = None
        for order, model in enumerate(models):
            if i >= order:
                ngram = data[i-(order):i+1]
                cond  = ngram[:-1]

                denom = len_train if cond == '' else models[order-1][cond]
                lprob = np.log2(models[order][ngram] + smoothing[order]) - np.log2(denom + smoothing[order] * numtokens)

            res[order] += - lprob

    return res

# Function to build and evaluate Markov models
def markov(train, val, test, max_order=3, lambdas=[1.0, 0.1 , 0.01, 0.0001, 1e-6], numtokens=None, verbose=False):
    ran = tqdm.trange if verbose else range

    models = [Counter() for o in range(max_order + 1)]

    if verbose:
        print('Creating frequency models.')
        tic()

    for i in ran(len(train)):
        for order, model in enumerate(models):
            if i >= order:
                ngram = train[i-(order):i+1]
                assert len(ngram) == order + 1, f'{i=}, {order=}'
                model[ngram] += 1

    if verbose: 
        print(f'done ({toc():.4}s).')

    numtokens = len(models[0]) if numtokens is None else numtokens

    if verbose: 
        print('Choosing smoothing levels.')
    res_val = []

    for i in ran(len(lambdas)):
        l = lambdas[i]
        res = codelength(models, val, len(train), numtokens=numtokens, smoothing=l, verbose=False)
        res_val.append(res)

    matrix = torch.tensor(res_val)
    lambda_indices = matrix.argmin(dim=0)
    smoothing = [lambdas[i] for i in lambda_indices]
    if verbose: 
        print('smoothing levels chosen: ', smoothing)

    if verbose:
        print('Computing codelengths.')
        tic()

    res = codelength(models, test, len(train), numtokens=numtokens, smoothing=smoothing, verbose=verbose)

    if verbose: 
        print(f'done ({toc():.4}s).')

    res = [r / len(test) for r in res]
    return models, res

# Function to save the model
def save_model(models, filename):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Model saved to {filename}")

# Main script
def main():
    data_path = "data/enwik8.gz"  # Adjust the path as necessary
    model_filename = "markov_model.pkl"  # File to save the model

    data_train, data_val, data_test = enwik8(data_path)

    # Flatten the data
    train_data = data_train.numpy().tobytes().decode('utf-8', errors='ignore')
    val_data = data_val.numpy().tobytes().decode('utf-8', errors='ignore')
    test_data = data_test.numpy().tobytes().decode('utf-8', errors='ignore')

    # Compute the compression length using Markov models
    models, compression_lengths = markov(train_data, val_data, test_data, max_order=5, verbose=True)
    print("Compression lengths:", compression_lengths)

    # Save the model
    save_model(models, model_filename)

if __name__ == "__main__":
    main()
