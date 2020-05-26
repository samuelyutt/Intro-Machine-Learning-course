import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import csv
import math
import random
import statistics
import matplotlib.pyplot as plt

with open('HW3_data/Logistic_data1-1.txt', newline='') as csvfile:
    datas11 = list(csv.reader(csvfile))
    
with open('HW3_data/Logistic_data1-2.txt', newline='') as csvfile:
    datas12 = list(csv.reader(csvfile))
    
with open('HW3_data/Logistic_data2-1.txt', newline='') as csvfile:
    datas21 = list(csv.reader(csvfile))
    
with open('HW3_data/Logistic_data2-2.txt', newline='') as csvfile:
    datas22 = list(csv.reader(csvfile))

#### Settings ####
# Set method
# 1: L2-norm
# 2: Cross Entropy
method = 1

savefig = 0

weight_traing_count_1 = 100
weight_traing_count_2 = 1000

alpha = 0.25

#### Functions ####
def innerprdt(a, b):
    prdt = 0;
    for i in range( len(a) ):
        prdt += (a[i] * b[i])
    return prdt

def sigmoid(wt, x):
    try:
        num = math.exp( -innerprdt(wt, x) )
    except OverflowError:
        num = float('inf')
    return 1/(1+num)


print("Original data")
xs = []
ys = []
for row in datas11:
    xs.append(float(row[0]))
    ys.append(float(row[1]))
plt.plot(xs, ys, 'ro')

xs = []
ys = []
for row in datas12:
    xs.append(float(row[0]))
    ys.append(float(row[1]))
plt.plot(xs, ys, 'bo')
if savefig:
    plt.savefig("Outputs/2_input1.png")
plt.show()

w = []
for i in range( len(datas11[0])+1 ):
    w.append(0)

for c in range(weight_traing_count_1):
    vxs = []
    for data in datas11:
        vx = [1]
        vx.append(float(data[0]))
        vx.append(float(data[1]))
        vxs.append(vx)

    for i in range( len(vxs) ):
        wp = []
        for j in range( len(vxs[i]) ):
            wp.append(0)
        for j in range( len(vxs[i]) ):
            if method == 1:
                wp[j] += (0 - sigmoid(w, vxs[i])) * sigmoid(w, vxs[i]) * (1 - sigmoid(w, vxs[i])) * vxs[i][j]
            elif method == 2:
                wp[j] += (sigmoid(w, vxs[i]) - 0) * vxs[i][j]
        if method == 2:
            for j in range( len(vxs[i]) ):
                wp[j] /= -(len(datas11) + len(datas12))
        for j in range( len(wp) ):
            w[j] += alpha * wp[j]

    vxs = []
    for data in datas12:
        vx = [1]
        vx.append(float(data[0]))
        vx.append(float(data[1]))
        vxs.append(vx)

    for i in range( len(vxs) ):
        wp = []
        for j in range( len(vxs[i]) ):
            wp.append(0)
        for j in range( len(vxs[i]) ):
            if method == 1:
                wp[j] += (1 - sigmoid(w, vxs[i])) * sigmoid(w, vxs[i]) * (1 - sigmoid(w, vxs[i])) * vxs[i][j]
            elif method == 2:
                wp[j] += (sigmoid(w, vxs[i]) - 1) * vxs[i][j]
        if method == 2:
            for j in range( len(vxs[i]) ):
                wp[j] /= -(len(datas11) + len(datas12))
        for j in range( len(wp) ):
            w[j] += alpha * wp[j]

xs1 = []
ys1 = []
xs0 = []
ys0 = []
tp = 0
tn = 0
fp = 0
fn = 0

for data in datas11:
    vx = [1]
    vx.append(float(data[0]))
    vx.append(float(data[1]))
    
    if sigmoid(w, vx) < 0.5:
        xs0.append(vx[1])
        ys0.append(vx[2])
        tn += 1
    else:
        xs1.append(vx[1])
        ys1.append(vx[2])
        fp += 1

for data in datas12:
    vx = [1]
    vx.append(float(data[0]))
    vx.append(float(data[1]))
    
    if sigmoid(w, vx) < 0.5:
        xs0.append(vx[1])
        ys0.append(vx[2])
        fn += 1
    else:
        xs1.append(vx[1])
        ys1.append(vx[2])
        tp += 1


plt.plot(xs0, ys0, 'ro')
plt.plot(xs1, ys1, 'bo')
if savefig:
    plt.savefig("Outputs/2_output1_method="+str(method)+".png")
plt.show()

print("--------------------------------------------------")
print("Method              ", method)
print("Weight              ", w)
print("--------------------------------------------------")           
print("Confusion matrix")
print("                     Is cluster 1        Is cluster 2")
print("Predict cluster 1   ", tn, "                 ", fn)
print("Predict cluster 2   ", fp, "                 ", tp)
print("--------------------------------------------------")
print("Precision           ", (tp) / (tp+fp))
print("Recall              ", (tp) / (tp+fn))


print("Original data")
xs = []
ys = []
for row in datas21:
    xs.append(float(row[0]))
    ys.append(float(row[1]))
plt.plot(xs, ys, 'ro')

xs = []
ys = []
for row in datas22:
    xs.append(float(row[0]))
    ys.append(float(row[1]))
plt.plot(xs, ys, 'bo')
if savefig:
    plt.savefig("Outputs/2_input2.png")
plt.show()

w = []
for i in range( len(datas21[0])+1 ):
    w.append(0)

for c in range(weight_traing_count_1):
    vxs = []
    for data in datas21:
        vx = [1]
        vx.append(float(data[0]))
        vx.append(float(data[1]))
        vxs.append(vx)

    for i in range( len(vxs) ):
        wp = []
        for j in range( len(vxs[i]) ):
            wp.append(0)
        for j in range( len(vxs[i]) ):
            if method == 1:
                wp[j] += (0 - sigmoid(w, vxs[i])) * sigmoid(w, vxs[i]) * (1 - sigmoid(w, vxs[i])) * vxs[i][j]
            elif method == 2:
                wp[j] += (sigmoid(w, vxs[i]) - 0) * vxs[i][j]
        if method == 2:
            for j in range( len(vxs[i]) ):
                wp[j] /= -(len(datas21) + len(datas22))
        for j in range( len(wp) ):
            w[j] += alpha * wp[j]

    vxs = []
    for data in datas22:
        vx = [1]
        vx.append(float(data[0]))
        vx.append(float(data[1]))
        vxs.append(vx)

    for i in range( len(vxs) ):
        wp = []
        for j in range( len(vxs[i]) ):
            wp.append(0)
        for j in range( len(vxs[i]) ):
            if method == 1:
                wp[j] += (1 - sigmoid(w, vxs[i])) * sigmoid(w, vxs[i]) * (1 - sigmoid(w, vxs[i])) * vxs[i][j]
            elif method == 2:
                wp[j] += (sigmoid(w, vxs[i]) - 1) * vxs[i][j]
        if method == 2:
            for j in range( len(vxs[i]) ):
                wp[j] /= -(len(datas21) + len(datas22))
        for j in range( len(wp) ):
            w[j] += alpha * wp[j]

xs1 = []
ys1 = []
xs0 = []
ys0 = []
tp = 0
tn = 0
fp = 0
fn = 0

for data in datas21:
    vx = [1]
    vx.append(float(data[0]))
    vx.append(float(data[1]))
    
    if sigmoid(w, vx) < 0.5:
        xs0.append(vx[1])
        ys0.append(vx[2])
        tn += 1
    else:
        xs1.append(vx[1])
        ys1.append(vx[2])
        fp += 1

for data in datas22:
    vx = [1]
    vx.append(float(data[0]))
    vx.append(float(data[1]))
    
    if sigmoid(w, vx) < 0.5:
        xs0.append(vx[1])
        ys0.append(vx[2])
        fn += 1
    else:
        xs1.append(vx[1])
        ys1.append(vx[2])
        tp += 1


plt.plot(xs0, ys0, 'ro')
plt.plot(xs1, ys1, 'bo')
if savefig:
    plt.savefig("Outputs/2_output2_method="+str(method)+".png")
plt.show()

print("--------------------------------------------------")
print("Method              ", method)
print("Weight              ", w)
print("--------------------------------------------------")           
print("Confusion matrix")
print("                     Is cluster 1         Is cluster 2")
print("Predict cluster 1   ", tn, "                 ", fn)
print("Predict cluster 2   ", fp, "                 ", tp)
print("--------------------------------------------------")
print("Precision           ", (tp) / (tp+fp))
print("Recall              ", (tp) / (tp+fn))