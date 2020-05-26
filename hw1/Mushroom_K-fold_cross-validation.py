#K-fold cross-validation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import csv
import math
import random
import statistics

feature_count =  [2, 6, 4, 10, 2, 9, 4, 3, 2, 12, 2, 6, 4, 4, 9, 9, 2, 4, 3, 8, 9, 6, 7]
k = 3

with open('Downloads/agaricus-lepiota.data', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    random.shuffle(rows)
    
    
    feature_list = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    data_count = 0
    train_data_count = [0, 0, 0]
    test_data_count = [0, 0, 0]
    TE = [0, 0, 0]
    FE = [0, 0, 0]
    FP = [0, 0, 0]
    TP = [0, 0, 0]
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if row[11] is '?':
            del rows[r]
    data_count = len(rows)
    
    test_range_e = 0
    k_ford_k = 3
    for k_ford_i in range(k_ford_k):
        test_range_s = test_range_e
        test_range_e = test_range_s + int(data_count / k_ford_k)
        test_range_e = min(test_range_e, data_count)

        # Train
        for r in range (0, data_count):
            if r in range (test_range_s, test_range_e):
                continue

            row = rows[r]

            label = row[0]
            if label in feature_list[0]:
                feature_list[0][label]+=1
            else:
                feature_list[0][label] = 1

            for i in range (1, 23):
                label_feature = label + row[i]
                if label_feature in feature_list[i]:
                    feature_list[i][label_feature]+=1
                else:
                    feature_list[i][label_feature] = 1
            train_data_count[k_ford_i]+=1

        # Test
        for r in range (test_range_s, test_range_e):
            row = rows[r]

            p_edible = 0
            p_poisonous = 0
            for i in range (1, 23):
                if 'e'+row[i] in feature_list[i]:
                    p_edible += math.log( ( feature_list[i]['e'+row[i]] + k ) / ( feature_list[0]['e'] + (k*feature_count[i]) ) )
                elif k is not 0:
                    p_edible += math.log(1/feature_count[i])
                if 'p'+row[i] in feature_list[i]:
                    p_poisonous += math.log( ( feature_list[i]['p'+row[i]] + k ) / ( feature_list[0]['p']+ (k*feature_count[i]) ) )
                elif k is not 0:
                    p_poisonous += math.log(1/feature_count[i])

            p_edible += math.log( feature_list[0]['e'] / (feature_list[0]['e'] + feature_list[0]['p']) )
            p_poisonous += math.log( feature_list[0]['p'] / (feature_list[0]['e'] + feature_list[0]['p']) )

            if p_edible >= p_poisonous:
                # Predict Edible
                if row[0] is 'e':
                    TE[k_ford_i]+=1
                elif row[0] is 'p':
                    FP[k_ford_i]+=1
            else:
                # Predict Poisonous
                if row[0] is 'e':
                    FE[k_ford_i]+=1
                elif row[0] is 'p':
                    TP[k_ford_i]+=1

            test_data_count[k_ford_i]+=1
    
# Results
print("K-fold cross-validation")
print("--------------------------------------------------")
print("Confusion matrix")
print("                 Predict Edible               Predict Poisonous")
print("Actual Edible    ", statistics.mean(TE), "              ", statistics.mean(FE))
print("Actual Poisonous ", statistics.mean(FP), "              ", statistics.mean(TP))
print("--------------------------------------------------")
print("Accuracy              ", (statistics.mean(TE)+statistics.mean(TP)) / (statistics.mean(TE)+statistics.mean(TP)+statistics.mean(FE)+statistics.mean(FP)))
print("Precision (Edible)    ", (statistics.mean(TE)) / (statistics.mean(TE)+statistics.mean(FE)))
print("Precision (Poisonous) ", (statistics.mean(TP)) / (statistics.mean(TP)+statistics.mean(FP)))
print("Recall (Edible)       ", (statistics.mean(TE)) / (statistics.mean(TE)+statistics.mean(FP)))
print("Recall (Poisonous)    ", (statistics.mean(TP)) / (statistics.mean(TP)+statistics.mean(FE)))
print("--------------------------------------------------")
print("data_count             ", data_count)
print("train_data_count (avg) ", statistics.mean(train_data_count))
print("test_data_count (avg)  ", statistics.mean(test_data_count))
print("Laplace k              ", k)

###
#                   |   Predict  |    Predict
#                   |   Edible   |   Poisonous
#-------------------+------------+-------------
#  Actual Edible    |     TE     |     FE
#-------------------+------------+-------------
#  Actual Poisonous |     FP     |     TP
#
###
        