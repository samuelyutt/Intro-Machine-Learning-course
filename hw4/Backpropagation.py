import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import csv
import math
import random
import statistics
import matplotlib.pyplot as plt

with open('src/data.txt', newline='') as csvfile:
    datas = list(csv.reader(csvfile))

train_count = 1000
accuracy = []
savefig = 0
network = [ [ {'weights': [0.5, 0.2, 0.1]}, {'weights': [0.7, 0.3, 0.4]} ], [ {'weights': [0.8, 0.1, 0.6]} ], [ {'weights': [0.9, 0.3]} ] ]

def sigmoid(x):
    try:
        num = 1.0 / (1.0 + np.exp(-x))
    except OverflowError:
        num = 0
    return num

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def forward(data):
    inputs = data;
    for layer in network:
        next_inputs = []
        for neuron in layer:
            value = 0.0
            for i in range ( len(neuron['weights'])-1 ):
                value += neuron['weights'][i] * inputs[i]
            value += neuron['weights'][ len(neuron['weights'])-1 ]
            
            value = sigmoid(value)
            neuron['value'] = value        
            next_inputs.append(value)
            
        inputs = next_inputs
    return inputs

def back(label):
    for i in range(len(network)-1, -1, -1):
        layer = network[i]
        deltas = []
        
        if i == len(network)-1:
            # Output layers
            neuron = layer[0]
            error = label - neuron['value']
            delta = error * sigmoid_deriv(neuron['value'])
            deltas.append(delta)
            
        else:
            # Hidden layers
            for j in range( len(layer) ):
                error = 0.0
                for neuron in network[ i+1 ]:
                    error += (neuron['weights'][j] * neuron['delta'])
                delta = error * sigmoid_deriv(neuron['value'])
                deltas.append(delta)
		
        for j in range( len(layer) ):
            layer[j]['delta'] = deltas[j]

def update(data):
    for i in range( len(network) ):
        inputs = []
        
        if i != 0:
            for neuron in network[ i-1 ]:
                inputs.append(neuron['value'])
        else:
            inputs = data
        
        for neuron in network[i]:
            for j in range( len(inputs) ):
                neuron['weights'][j] += neuron['delta'] * inputs[j]
            neuron['weights'][ len(neuron['weights'])-1 ] += neuron['delta']

def cal_accuracy(test_range_s, test_range_e):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for r in range(test_range_s, test_range_e):
        # Test
        data = []
        data.append(float(datas[r][0]))
        data.append(float(datas[r][1]))
        data.append(float(datas[r][2]))

        # Predict
        outputs = forward(data)
        pdct = 0
        if (outputs[0] > 0.5):
            pdct = 1
        else:
            pdct = 0

        if pdct == 1:
            if (int(datas[r][2]) == 1):
                tp += 1
            else:
                fp += 1
        else:
            if (int(datas[r][2]) == 1):
                fn += 1
            else:
                tn += 1
    accuracy.append((tp+tn)/(tp+tn+fp+fn))

random.shuffle(datas)

data_count = len(datas)

random.shuffle(datas)
test_range_s = 0
test_range_e = test_range_s + int(data_count * 0.3)

for epoch in range(train_count+1):
    loss = 0.0
    for r in range (0, data_count):
        # Train
        data = []
        data.append(float(datas[r][0]))
        data.append(float(datas[r][1]))
        data.append(float(datas[r][2]))
        
        # Forward propagation
        output = forward(data)
        
        # Calculate loss
        loss += (float(datas[r][2]) - output[0]) * (float(datas[r][2]) - output[0])
        
        # Back propagation
        back(float(datas[r][2]))
        
        # Update weights
        update(data)
    
    # Print loss
    if (epoch % (train_count/10) == 0 and epoch != 0):
        print("epochs", epoch, "loss:", loss)
        cal_accuracy(0, data_count)

test_range_s = 0
test_range_e = data_count

pdct1x = []
pdct1y = []
pdct0x = []
pdct0y = []

trth1x = []
trth1y = []
trth0x = []
trth0y = []
tp = 0
tn = 0
fp = 0
fn = 0

for r in range(test_range_s, test_range_e):
    # Test
    data = []
    data.append(float(datas[r][0]))
    data.append(float(datas[r][1]))
    data.append(float(datas[r][2]))
    
    # Predict
    outputs = forward(data)
    pdct = 0
    if (outputs[0] > 0.5):
        pdct = 1
    else:
        pdct = 0
    
    if pdct == 1:
        pdct1x.append(float(datas[r][0]))
        pdct1y.append(float(datas[r][1]))
        if (int(datas[r][2]) == 1):
            tp += 1
            trth1x.append(float(datas[r][0]))
            trth1y.append(float(datas[r][1]))
        else:
            fp += 1
            trth0x.append(float(datas[r][0]))
            trth0y.append(float(datas[r][1]))
    else:
        pdct0x.append(float(datas[r][0]))
        pdct0y.append(float(datas[r][1]))
        if (int(datas[r][2]) == 1):
            fn += 1
            trth1x.append(float(datas[r][0]))
            trth1y.append(float(datas[r][1]))
        else:
            tn += 1
            trth0x.append(float(datas[r][0]))
            trth0y.append(float(datas[r][1]))


# Plot predictions and ground truth
x = np.linspace(0, 100, 256, endpoint = True)
y = 0;
tempX = 1
for i in range(1):
    y = x    
    
plt.title("Predict result")
plt.plot(x, y, 'g')
plt.plot(pdct0x, pdct0y, 'ro')
plt.plot(pdct1x, pdct1y, 'bs')
if savefig:
    plt.savefig("Outputs/Predict_result.png")
plt.show()

plt.title("Ground truth")
plt.plot(x, y, 'g')
plt.plot(trth0x, trth0y, 'ro')
plt.plot(trth1x, trth1y, 'bs')
if savefig:
    plt.savefig("Outputs/Ground_truth.png")
plt.show()

# Print accuracy
print("--------------------------------------------------") 
print("Accuracy            ", accuracy)

# Print final results
print("--------------------------------------------------") 
print("Holdout Validation")
print("--------------------------------------------------")           
print("Confusion matrix")
print("                     Predict positive    Predict negative")
print("Actual positive     ", tp, "                 ", fn)
print("Actual negative     ", fp, "                 ", tn)
print("--------------------------------------------------")
print("Accuracy            ", (tp+tn)/(tp+tn+fp+fn))
print("Sensitivity (True)  ", (tp) / (tp+fn))
print("Sensitivity (False) ", (tn) / (fp+tn))
print("Precision (True)    ", (tp) / (tp+fp))
print("Precision (False)   ", (tn) / (tn+fn))