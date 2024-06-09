import pickle
import numpy as np
from collections import Counter

# Function to load the model
def load_model(filename):
    with open(filename, 'rb') as f:
        models = pickle.load(f)
    print(f"Model loaded from {filename}")
    return models

# Function to get probabilities of the next symbol
def get_next_symbol_probabilities(models, context, numtokens, smoothing):
    order = 4  # We are using the 4th order model
    if len(context) < order:
        raise ValueError(f"Context length must be at least {order} for 4th order prediction")

    ngram = context[-order:]
    model = models[order]
    counts = np.array([model[ngram + chr(i)] for i in range(numtokens)])
    probabilities = (counts + smoothing[order]) / (sum(counts) + smoothing[order] * numtokens)
    
    return probabilities

# Function to predict the next symbol
def predict_next_symbol(probabilities):
    return np.argmax(probabilities)

# Main script
def main():
    model_filename = "markov_model.pkl"  # File to load the model
    numtokens = 256  # Number of tokens in the model
    smoothing = [1.0, 0.1, 0.01, 0.0001, 0.00001]  # Smoothing parameters for different orders

    # Load the model
    models = load_model(model_filename)

    # Example context
    context = " fox"
    
    if len(context) < 4:
        raise ValueError("Context length must be at least 4 for 4th order prediction")

    # Get the probabilities for the next symbol
    probabilities = get_next_symbol_probabilities(models, context, numtokens, smoothing)
    print("Probabilities for the next symbol:", probabilities)

    # Predict the next symbol
    next_symbol_index = predict_next_symbol(probabilities)
    next_symbol = chr(next_symbol_index)
    print("Next predicted symbol:", next_symbol)

if __name__ == "__main__":
    main()
