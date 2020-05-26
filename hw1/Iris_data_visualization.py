import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import csv
import math
import random
import statistics

savefig = 0

def GD(x, mu, sigma):
    N = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return(N)

with open('bezdekIris.data', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    feature = [ [ [], [], [], [] ], [ [], [], [], [] ], [ [], [], [], [] ], [ [], [], [], [] ] ]
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if len(row) is 0:
            del rows[r]
    data_count = len(rows)
    
    for r in range (0, data_count):
        row = rows[r]
        
        iris = 0
        if row[4] == 'Iris-setosa':
            iris = 0
        elif row[4] == 'Iris-versicolor':
            iris = 1
        elif row[4] == 'Iris-virginica':
            iris = 2
        
        for f in range(4):
            feature[f][iris].append(float(row[f]))
            feature[f][3].append(float(row[f]))
    
    feature_name = ["sepal length", "sepal width", "petal length", "petal width"]
    condition_name = ["|setosa", "|versicolor", "|virginica", ""]
    
    for f in range(4):
        for c in range(4):
            print("Average of", feature_name[f]+condition_name[c], statistics.mean(feature[f][c]))
            print("Standard deviation of", feature_name[f]+condition_name[c], statistics.stdev(feature[f][c]))
            plt.hist(feature[f][c])
            plt.title(feature_name[f]+condition_name[c])
            plt.ylabel ('Count')
            plt.xlabel ("Value")
            if savefig:
                plt.savefig("Visualization/Iris/"+feature_name[f]+condition_name[c]+" distribution.png")
            #plt.show()
    
    