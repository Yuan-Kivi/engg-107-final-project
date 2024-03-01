import numpy as np
import pandas as pd
import pymc3 as pm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import help as hlp
import torch.nn as nn
import torch.optim as optim
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import arviz as az
import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns  # Optional for enhanced visualization
import metrices


# Step 1: Dataset division
# load the dataset
file_path = './data/data.csv'
df = pd.read_csv(file_path)
df = df.iloc[:, 1:-1]
# Replace 'M' with 1 and 'B' with 0 in the 'target' column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.iloc[:, 1:]   # features
y = df.iloc[:, 0]    # labels

# divide to 80% training dataset and 20% testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = np.array(X_train, dtype=np.float64), np.array(X_test, dtype=np.float64), np.array(y_train, dtype=np.float64), np.array(y_test, dtype=np.float64)

X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test_norm = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)


# # Step 2: Bayesian method (MCMC)

iterations = 20000  # Adjust this number based on your convergence diagnostics
beta_init = np.zeros(X_train_norm.shape[1])  # Initialize beta values (parameter vector)

# Execute the M-H algorithm
samples = hlp.metropolis_hastings(X_train_norm, y_train, iterations, beta_init)

# Prediction using the average beta after burn-in
burn_in = int(iterations * 0.1)  # For example, 10% burn-in
beta_avg = np.mean(samples[burn_in:], axis=0)

# Test convergence
# Plot the trace for each parameter
# for i in range(X.shape[1]):
#     plt.plot(samples[:, i])
#     plt.title(f'Trace plot for parameter {i+1}')
#     plt.xlabel('Iterations')
#     plt.ylabel(f'Parameter {i+1} value')
#     plt.show()

# To use PyMC3/ArviZ effectively, your MCMC samples need to be in a format that these libraries can process (e.g.,
# xarray Dataset or InferenceData). The rhat statistic should ideally be close to 1 (values > 1.1 may indicate lack
# of convergence), and the effective sample size (ESS) should be large enough relative to your total sample size,
# indicating independent and informative samples.
# Let's create a dummy InferenceData object for demonstration;
# replace this with your actual data
inference_data = az.convert_to_inference_data(samples)
#
# R-hat diagnostics (Gelman-Rubin statistic)
rhat = az.rhat(inference_data)
print(rhat)

# Effective sample size
ess = az.ess(inference_data)
print(ess)
#
# # Plotting trace for visual inspection
# az.plot_trace(inference_data)
# plt.show()
#
# # Summary statistics for the MCMC samples
# summary = az.summary(inference_data)
# print(summary)


# Step 3: Machine Learning & Deep Learning methods
X_train_tensor = torch.FloatTensor(X_train_norm)
X_test_tensor = torch.FloatTensor(X_test_norm)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Step 3.1: MLP
model_mlp = models.MLP(input_size = X_train_tensor.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)

# Training the model
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model_mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # if epoch % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 3.2: SVM
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train_norm, y_train)

# Step 3.3: Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_norm, y_train)

# Step 4: Evaluation
pre_mcmc = hlp.predict(X_test_norm, beta_avg)
pre_mlp = np.where(model_mlp(X_test_tensor) > 0.5, 1, 0)
pre_svm = model_svm.predict(X_test_norm)
pre_dt = model_dt.predict(X_test_norm)

# Step 4: Accuracy, Recall, Precision

# Step 4.1.1 Confusion Matrix - Accuracy, Recall, Precision
metrices.Confusion_Matrix(y_true=y_test, y_pred=pre_mcmc, model='MCMC - Metropolis-Hastings Algorithm')
metrices.Confusion_Matrix(y_true=y_test, y_pred=pre_mlp, model='Multilayer Perceptron (MLP)')
metrices.Confusion_Matrix(y_true=y_test, y_pred=pre_svm, model='Supporting Vector Machine (SVM)')
metrices.Confusion_Matrix(y_true=y_test, y_pred=pre_dt, model='Decision Tree (DT)')

# Step 5: Sensitivity analysis

# import numpy as np
# from SALib.sample import saltelli
# from SALib.analyze import sobol
#
# # Define the problem
# problem = {
#     'num_vars': 29,  # Let's just analyze proposal_width for simplicity
#     'names': ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x2','x26','x27','x28','x29'],
#     # 'names': ['proposal_width'],
#     'bounds': [[0, 1]]  # Adjust based on reasonable values for your context
# }
#
# # Generate samples for proposal_width
# param_values = saltelli.sample(problem, 32)
#
# # Placeholder for MCMC outputs
# # Assume we're interested in the mean of some posterior parameter for simplicity
# output_means = np.zeros(param_values.shape[0])
#
# # Run the MCMC simulation varying the proposal_width
# for i, params in enumerate(param_values):
#     # Unpack your parameters (just proposal_width in this case)
#     proposal_width = params[0]
#
#     # Run your M-H algorithm (assuming it returns the sample array)
#     # You need to adapt your metropolis_hastings function to accept proposal_width as an argument
#     samples = hlp.metropolis_hastings(X_train_norm, y_train, iterations, beta_init, proposal_width=proposal_width)
#
#     # Compute the mean of the first parameter for simplicity
#     output_means[i] = np.mean(samples[:, 0])
#     print(i)
#
# # Analyze the results using Sobol sensitivity analysis
# Si = sobol.analyze(problem, output_means, print_to_console=False)
#
# # Output the Sobol indices
# print("First-order Sobol index:", Si['S1'])
# print("Total-effect Sobol index:", Si['ST'])
#
# ##############
#
# # Define a range of proposal widths to test
# proposal_widths = np.linspace(0.01, 1.0, 10)
#
# # Placeholder for storing summary statistics for each proposal width
# means = []
# variances = []
#
# for width in proposal_widths:
#     # Run the MCMC algorithm
#     samples = hlp.metropolis_hastings(X_train_norm, y_train, iterations, beta_init, proposal_width=width)
#
#     # Calculate the posterior summary statistics after burn-in
#     post_burn_in_samples = samples[int(iterations * 0.1):]
#     mean = np.mean(post_burn_in_samples, axis=0)
#     variance = np.var(post_burn_in_samples, axis=0)
#
#     # Store the results
#     means.append(mean)
#     variances.append(variance)

# # Plot the sensitivity of the posterior mean to the proposal width
# plt.plot(proposal_widths, means)
# plt.xlabel('Proposal Width')
# plt.ylabel('Posterior Mean')
# plt.title('Sensitivity of Posterior Mean to Proposal Width')
# plt.legend()
# plt.show()
#
# # Plot the sensitivity of the posterior variance to the proposal width
# plt.plot(proposal_widths, variances)
# plt.xlabel('Proposal Width')
# plt.ylabel('Posterior Variance')
# plt.title('Sensitivity of Posterior Variance to Proposal Width')
# plt.show()
# #
# # proposal_widths = np.linspace(0.01, 1.0, len(means[0]))
# means = np.array(means)
# for i in range(means.shape[1]):
#     plt.plot(proposal_widths, means[:, i], label=f'Parameter {i+1}')
# plt.xlabel('Proposal Width')
# plt.ylabel('Posterior Mean')
# plt.title('Sensitivity of Posterior Mean to Proposal Width')
# plt.legend()  # This will add the legend to your plot
# plt.show()
#
# variances = np.array(variances)
# # Plot the sensitivity of the posterior variance to the proposal width
# for i in range(variances.shape[1]):
#     plt.plot(proposal_widths, variances[:, i], label=f'Parameter {i+1}')
# plt.xlabel('Proposal Width')
# plt.ylabel('Posterior Variance')
# plt.title('Sensitivity of Posterior Variance to Proposal Width')
# plt.legend()  # This will add the legend to your plot
# plt.show()

# 对parameter space做softmax

sen_ana = hlp.softmax(beta_avg)

sorted_indices = np.argsort(sen_ana)[::-1]
print(sorted_indices)
sorted_sensitivities = sen_ana[sorted_indices]
print(sorted_sensitivities)
sorted_labels = [f'Parameter {i+1}' for i in sorted_indices]

colors = [
    'lightblue', 'lightcoral', 'lightgreen', 'lightskyblue', 'lightpink',
    'yellowgreen', 'orange', 'gold', 'purple', 'violet',
    # Add more colors to match the number of parameters
]

# Create the pie chart
plt.figure(figsize=(10, 10))
plt.pie(sorted_sensitivities, startangle=140, colors=colors)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sensitivity Analysis Pie Chart (Sorted)')
plt.show()