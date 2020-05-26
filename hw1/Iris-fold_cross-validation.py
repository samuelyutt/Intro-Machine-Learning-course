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


def GD(x, mu, sigma):
    N = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return(N)

with open('bezdekIris.data', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    random.shuffle(rows)

    data_count = 0
    train_data_count = [0, 0, 0]
    test_data_count = [0, 0, 0]
    setosa = [0, 0, 0]
    versicolor = [0, 0, 0]
    virginica = [0, 0, 0]
    TS = [0, 0, 0]
    FSC = [0, 0, 0]
    FSG = [0, 0, 0]
    TG = [0, 0, 0]
    FGS = [0, 0, 0]
    FGC = [0, 0, 0]
    TC = [0, 0, 0]
    FCS = [0, 0, 0]
    FCG = [0, 0, 0]
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if len(row) is 0:
            del rows[r]
    data_count = len(rows)
    
    test_range_e = 0
    k_ford_k = 3
    for k_ford_i in range(k_ford_k):
        test_range_s = test_range_e
        test_range_e = test_range_s + int(data_count / k_ford_k)
        test_range_e = min(test_range_e, data_count)
        
        sl = [ [], [], [] ]
        sw = [ [], [], [] ]
        pl = [ [], [], [] ]
        pw = [ [], [], [] ]
        
        # Train
        for r in range (0, data_count):
            if r in range (test_range_s, test_range_e):
                continue

            row = rows[r]

            iris = 0
            if row[4] == 'Iris-setosa':
                iris = 0
                setosa[k_ford_i]+=1
            elif row[4] == 'Iris-versicolor':
                iris = 1
                versicolor[k_ford_i]+=1
            elif row[4] == 'Iris-virginica':
                iris = 2
                virginica[k_ford_i]+=1

            sl[iris].append(float(row[0]))
            sw[iris].append(float(row[1]))
            pl[iris].append(float(row[2]))
            pw[iris].append(float(row[3]))

            train_data_count[k_ford_i]+=1

        # Test
        for r in range(test_range_s, test_range_e):
            row = rows[r]

            p_setosa = 0
            p_versicolor = 0
            p_virginica = 0

            p_setosa += math.log( GD(float(row[0]), statistics.mean(sl[0]), statistics.stdev(sl[0])) )
            p_setosa += math.log( GD(float(row[1]), statistics.mean(sw[0]), statistics.stdev(sw[0])) )
            p_setosa += math.log( GD(float(row[2]), statistics.mean(pl[0]), statistics.stdev(pl[0])) )
            p_setosa += math.log( GD(float(row[3]), statistics.mean(pw[0]), statistics.stdev(pw[0])) )
            p_setosa += math.log( setosa[k_ford_i] / (setosa[k_ford_i]+versicolor[k_ford_i]+virginica[k_ford_i]) )

            p_versicolor += math.log( GD(float(row[0]), statistics.mean(sl[1]), statistics.stdev(sl[1])) )
            p_versicolor += math.log( GD(float(row[1]), statistics.mean(sw[1]), statistics.stdev(sw[1])) )
            p_versicolor += math.log( GD(float(row[2]), statistics.mean(pl[1]), statistics.stdev(pl[1])) )
            p_versicolor += math.log( GD(float(row[3]), statistics.mean(pw[1]), statistics.stdev(pw[1])) )
            p_versicolor += math.log( versicolor[k_ford_i] / (setosa[k_ford_i]+versicolor[k_ford_i]+virginica[k_ford_i]) )

            p_virginica += math.log( GD(float(row[0]), statistics.mean(sl[2]), statistics.stdev(sl[2])) )
            p_virginica += math.log( GD(float(row[1]), statistics.mean(sw[2]), statistics.stdev(sw[2])) )
            p_virginica += math.log( GD(float(row[2]), statistics.mean(pl[2]), statistics.stdev(pl[2])) )
            p_virginica += math.log( GD(float(row[3]), statistics.mean(pw[2]), statistics.stdev(pw[2])) )
            p_virginica += math.log( virginica[k_ford_i] / (setosa[k_ford_i]+versicolor[k_ford_i]+virginica[k_ford_i]) )

            max_p = max(p_setosa, p_versicolor, p_virginica)

            if max_p == p_setosa:
                # Predict setosa
                if row[4] == 'Iris-setosa':
                    TS[k_ford_i]+=1
                elif row[4] == 'Iris-versicolor':
                    FSC[k_ford_i]+=1
                elif row[4] == 'Iris-virginica':
                    FSG[k_ford_i]+=1
            elif max_p == p_versicolor:
                # Predict versicolor
                if row[4] == 'Iris-setosa':
                    FCS[k_ford_i]+=1
                elif row[4] == 'Iris-versicolor':
                    TC[k_ford_i]+=1
                elif row[4] == 'Iris-virginica':
                    FCG[k_ford_i]+=1
            elif max_p == p_virginica:
                # Predict virginica
                if row[4] == 'Iris-setosa':
                    FGS[k_ford_i]+=1
                elif row[4] == 'Iris-versicolor':
                    FGC[k_ford_i]+=1
                elif row[4] == 'Iris-virginica':
                    TG[k_ford_i]+=1

            test_data_count[k_ford_i]+=1
    
# Results
print("K-fold cross-validation")
print("-----------------------------------------------------------------------")
print("Confusion matrix")
print("                  Predict setosa      Predict versicolor      Predict virginica")
print("Actual setosa     ", statistics.mean(TS), "                ", statistics.mean(FSC), "                     ", statistics.mean(FSG))
print("Actual versicolor ", statistics.mean(FGS), "                ", statistics.mean(TC), "                     ", statistics.mean(FCG))
print("Actual virginica  ", statistics.mean(FGS), "                ", statistics.mean(FGC), "                     ", statistics.mean(TG))
print("-----------------------------------------------------------------------")
print("Accuracy               ", (statistics.mean(TS)+statistics.mean(TC)+statistics.mean(TG)) / (statistics.mean(TS)+statistics.mean(TC)+statistics.mean(TG)+statistics.mean(FSC)+statistics.mean(FSG)+statistics.mean(FGS)+statistics.mean(FCG)+statistics.mean(FGS)+statistics.mean(FGC)))
print("Precision (setosa)     ", statistics.mean(TS) / (statistics.mean(TS)+statistics.mean(FSC)+statistics.mean(FSG)))
print("Precision (versicolor) ", statistics.mean(TC) / (statistics.mean(TC)+statistics.mean(FGS)+statistics.mean(FCG)))
print("Precision (virginica)  ", statistics.mean(TG) / (statistics.mean(TG)+statistics.mean(FGS)+statistics.mean(FGC)))
print("Recall (setosa)        ", statistics.mean(TS) / (statistics.mean(TS)+statistics.mean(FGS)+statistics.mean(FCG)+statistics.mean(FGS)+statistics.mean(FGC)))
print("Recall (versicolor)    ", statistics.mean(TC) / (statistics.mean(TC)+statistics.mean(FSC)+statistics.mean(FSG)+statistics.mean(FGS)+statistics.mean(FGC)))
print("Recall (virginica)     ", statistics.mean(TG) / (statistics.mean(TG)+statistics.mean(FSC)+statistics.mean(FSG)+statistics.mean(FGS)+statistics.mean(FCG)))
print("-----------------------------------------------------------------------")
print("data_count             ", data_count)
print("train_data_count (avg) ", statistics.mean(train_data_count))
print("test_data_count (avg)  ", statistics.mean(test_data_count))

###
#                    |   Predict  |   Predict   |   Predict
#                    |   setosa   |  versicolor |  virginica
#--------------------+------------+-------------+-------------
#  Actual setosa     |     TS     |     FSC     |     FSG
#--------------------+------------+-------------+-------------
#  Actual versicolor |     FCS    |     TC      |     FCG
#--------------------+------------+-------------+-------------
#  Actual virginica  |     FGS    |     FGC     |     TG
#
###