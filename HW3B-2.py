from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp, tanh
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

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
def initialize_network(n_inputs, n_hidden, n_outputs, initial_beta):
    network = list()
    if initial_beta == -1:
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        hidden_layer2 = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer2)
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
    else:
        hidden_layer = [{'weights': [np.random.normal(0, initial_beta) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        hidden_layer2 = [{'weights': [np.random.normal(0, initial_beta) for i in range(n_hidden + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer2)
        output_layer = [{'weights': [np.random.normal(0, initial_beta) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
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
        print('>epoch=%d, lrate=%.4f, error=%.3f' % (epoch, l_rate, sum_error))
        errorForPlot.append(sum_error)
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

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

errorForPlot = list()

# berry
filename = 'E:/study/201908/2IMM20/HW3A_python_example/HW3Atrain.csv'
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

leanring_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
beta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
accuracy = np.zeros((len(leanring_rate), len(beta)))
total_time = 0
for lr in range(len(leanring_rate)):
    for bt in range(len(beta)):
        firsttime = time.time()
        #for parameter beta, if wanna random initials set beta=-1
        network = initialize_network(n_inputs, 10, n_outputs, beta[bt])
        train_network(network, dataset, leanring_rate[lr], 2000, n_outputs)

        # Load test data
        filename = 'E:/study/201908/2IMM20/HW3A_python_example/HW3Avalidate.csv'
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

        for row in dataset:
            prediction = predict(network, row)
            if row[-1] == prediction:
                numberOfCorrectAnswers += 1
            else:
                numberOfWrongAnswers += 1
            # print('Expected=%d, Got=%d' % (row[-1], prediction))

        print('Correct: %d' % numberOfCorrectAnswers)
        print('Wrong: %d' % numberOfWrongAnswers)
        print('Total: %d' % len(dataset))

        secondtime = time.time()
        print('Execute time (seconds):          {:0.6f}'.format(secondtime - firsttime))
        total_time += secondtime - firsttime
        accuracy[lr][bt] = numberOfCorrectAnswers / len(dataset)
        iterationCount = len(errorForPlot)

        #plt.figure()
        #plt.plot(range(iterationCount), errorForPlot, "r-")
        #plt.xlabel("Iteration count")
        #plt.ylabel("Total error")
        #plt.show()
print('Total execute time (seconds): {:0.6f}'.format(total_time))
def draw_heatmap(data, ylabels, xlabels):
    cmap = cm.Blues
    figure = plt.figure(facecolor='w')
    ax = figure.add_subplot(111)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax = data[0][0]
    vmin = data[0][0]
    for i in data:
        for j in i:
            if j > vmax:
                vmax = j
            if j < vmin:
                vmin = j
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None)
    plt.xlabel("σ2")
    plt.ylabel("Learing rate")
    plt.title("Heatmap for accuracy")
    plt.show()

draw_heatmap(accuracy, leanring_rate, beta)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yticks(range(len(leanring_rate)))
ax.set_yticklabels(leanring_rate)
ax.set_xticks(range(len(beta)))
ax.set_xticklabels(beta)
im = ax.imshow(accuracy, cmap=plt.cm.summer_r)
plt.colorbar(im)
plt.xlabel("σ2")
plt.ylabel("Learing rate")
plt.title("Heatmap for accuracy")
plt.show()