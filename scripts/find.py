import torch
from torch import nn
import torch.nn.functional as F
from former import GTransformer
import numpy as np
import random
import gzip
from search import find_batch_size

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    print("Loading enwik8 dataset...")
    with gzip.open(path, 'rb') as file:
        data = file.read(n_train + n_valid + n_test)
        X = np.frombuffer(data, dtype=np.uint8).copy()
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def load_data_and_model(depth, embedding_size, num_heads, context, num_tokens, attention_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GTransformer(
        emb=embedding_size,
        heads=num_heads,
        depth=depth,
        seq_length=context,
        num_tokens=num_tokens,
        attention_type=attention_type,
    )
    model.to(device)
    dummy_input = torch.randint(0, num_tokens, (1, context)).to(device)
    # Define the dummy loss function, which uses the model's output
    dummy_loss = lambda output: F.nll_loss(output.transpose(2, 1), torch.randint(0, num_tokens, (1, context)).to(device))
    return model, dummy_input, dummy_loss

def main():
    depths = [3, 6, 9, 12]
    context = 256
    embedding_size = 768
    num_heads = 8
    num_tokens = 256
    attention_type = 'default'
    batch_sizes = {}

    for depth in depths:
        print(f"Evaluating optimal batch size for model depth: {depth}")
        model, dummy_input, dummy_loss = load_data_and_model(depth, embedding_size, num_heads, context, num_tokens, attention_type)
        optimal_batch_size = find_batch_size(model, dummy_loss, dummy_input)
        batch_sizes[depth] = optimal_batch_size
        print(f"Optimal batch size for depth {depth}: {optimal_batch_size}")

    print("Batch sizes for each depth:", batch_sizes)

if __name__ == "__main__":
    main()
