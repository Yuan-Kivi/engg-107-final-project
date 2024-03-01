import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def logistic(x):
    # return 1 / (1 + np.exp(-x))
    z = np.exp(x)
    return np.clip(z / (1 + z), -709.78, 709.78)

# Step 2: Implement Metropolis-Hastings for logistic regression
def metropolis_hastings(X, y, iterations, beta_init, proposal_width=0.1):
    # np.random.seed(12)
    n_params = X.shape[1]
    beta_current = beta_init
    samples = np.zeros((iterations, n_params))

    for i in range(iterations):
        # Propose new parameters
        beta_proposal = beta_current + np.random.normal(0, proposal_width, size=n_params)

        # Calculate likelihoods
        likelihood_current = np.prod(logistic(X @ beta_current) ** y * (1 - logistic(X @ beta_current)) ** (1 - y))
        likelihood_proposal = np.prod(logistic(X @ beta_proposal) ** y * (1 - logistic(X @ beta_proposal)) ** (1 - y))

        # Calculate prior probabilities (uniform prior)
        prior_current = 1
        prior_proposal = 1

        # Calculate acceptance probability
        p_accept = min(1, (likelihood_proposal * prior_proposal) / (likelihood_current * prior_current))

        # Accept or reject proposal
        accept = np.random.rand() < p_accept
        if accept:
            beta_current = beta_proposal

        samples[i] = beta_current

    return samples

def predict(X_new, beta_avg):
    prob = logistic(X_new @ beta_avg)
    # print(prob)
    matrix = np.where(prob > 0.5, 1, 0)
    return matrix

def accuracy(matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    matches = np.sum(matrix1 == matrix2)

    # Calculate the accuracy
    accuracy = matches / matrix1.shape[0]  # or use matrix2.size, they are the same

    # accuracy = (matrix1 == matrix2).float().mean()

    return accuracy

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum(axis=0)  # The axis=0 argument sums over the first dimension
