from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp, tanh
import numpy as np
import time
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    hidden_layer2 = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer2)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    goodResultCount = 0
    for epoch in range(n_epoch):
        sum_error = 0
        for m in range(80):
            randomindex = randrange(len(train))
            row = train[randomindex]
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errorForPlotTraining.append(sum_error)

        if sum_error < 1:
            goodResultCount += 1

        if goodResultCount > 2:
            break

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        # row[column] = row[column].replace('\\xef\\xbb\\xbf', '')
        # print('.', row[column], '.')
        # print('.', row[column].strip(), '.')
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


errorForPlotTraining = list()
firsttime = time.time()

# berry
# filename = 'HW3Atrain.csv'
filename = 'HW3Avalidate.csv'
dataset = load_csv(filename)
dataset.pop(0)

# convert string numbers to floats
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables to the range of 0 and 1 (range of the transfer function)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# Test training backprop algorithm
# seed(1)

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_inputs, n_outputs)
network = initialize_network(n_inputs, 10, n_outputs)
train_network(network, dataset, 0.5, 2000, n_outputs)
for layer in network:
    print(layer)


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

    # Test making predictions with the network


# berry
# filename = 'HW3Avalidate.csv'
filename = 'HW3Atrain.csv'
dataset = load_csv(filename)
dataset.pop(0)

# convert string numbers to floats
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables to the range of 0 and 1 (range of the transfer function)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


numberOfCorrectAnswers = 0
numberOfWrongAnswers = 0
predicted0ShouldBe1 = 0
predicted1ShouldBe0 = 0
predicted1ShouldBe1 = 0
predicted0ShouldBe0 = 0

for row in dataset:
    prediction = predict(network, row)
    if row[-1] == prediction:
        if row[-1] == 0:
            predicted0ShouldBe0 += 1
        else:
            predicted1ShouldBe1 += 1
        numberOfCorrectAnswers += 1
    else:
        if row[-1] == 0:
            predicted1ShouldBe0 += 1
        else:
            predicted0ShouldBe1 += 1
        numberOfWrongAnswers += 1
    # print('Expected=%d, Got=%d' % (row[-1], prediction))

print('Correct: %d' % numberOfCorrectAnswers)
print('Wrong: %d' % numberOfWrongAnswers)
print('Total: %d' % len(dataset))

print('predicted 1, should be 1', predicted1ShouldBe1)
print('predicted 0, should be 0', predicted0ShouldBe0)
print('predicted 1, should be 0', predicted1ShouldBe0)
print('predicted 0, should be 1', predicted0ShouldBe1)

secondtime = time.time()

print(secondtime - firsttime)

iterationCount = len(errorForPlotTraining)

plt.figure()
plt.plot(range(iterationCount), errorForPlotTraining, "r-")
plt.xlabel("Iteration count")
plt.ylabel("Total error")
plt.show()
