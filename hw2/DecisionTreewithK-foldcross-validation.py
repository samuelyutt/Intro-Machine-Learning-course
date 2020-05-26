import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import csv
import math
import random
import statistics

feature_dict = {}
ctgl_features = [2, 4, 6, 7, 8, 9, 10, 14]
tree = []
yrows = []
rows = []
data_count = 0;

def gini_index(data):
    v_count = len(data)
    t_count = 0
    
    if v_count == 0:
        return 0

    for i in range(0, v_count):
        if int(data[i][ len(data[i])-1 ]) == 1:
            t_count += 1

    f_count = v_count - t_count

    p_t = t_count / v_count
    p_f = f_count / v_count

    if (p_t) == 0:
        t_gini = 0
    else:
        t_gini = p_t * p_t

    if (p_f) == 0:
        f_gini = 0
    else:
        f_gini = p_f * p_f

    gini = 1 - ( t_gini + f_gini )
    
    return gini

def entropy(data):
    v_count = len(data)
    t_count = 0
    
    if v_count == 0:
        return 0

    for i in range(0, v_count):
        if int(data[i][ len(data[i])-1 ]) == 1:
            t_count += 1

    f_count = v_count - t_count

    p_t = t_count / v_count
    p_f = f_count / v_count

    if (p_t) == 0:
        t_entropy = 0
    else:
        t_entropy = p_t * math.log(p_t)

    if (p_f) == 0:
        f_entropy = 0
    else:
        f_entropy = p_f * math.log(p_f) 

    entropy = -( t_entropy + f_entropy )
    
    return entropy


def remainder(key, data):
    rmdr = 0
    for value in feature_dict[key]:
        v_count = 0
        sltd_data = []
        
        for i in range(0, len(data)):
            if data[i][key] == value:
                v_count += 1
                sltd_data.append(data[i])

        value = entropy(sltd_data)
        #value = gini_index(sltd_data)
        rmdr += value * ( v_count / len(data) )

    return rmdr


def remainder_cts(key, value, data):
    rmdr = 0
    
    b_count = 0
    s_count = 0
    b_data = []
    s_data = []
    for i in range(0, len(data)):
        if int(data[i][key]) < value:
            s_count += 1
            s_data.append(data[i])
        else:
            b_count += 1
            b_data.append(data[i])
    value = entropy(s_data)
    #value = gini_index(s_data)
    rmdr += value * ( s_count / len(data) )
    
    etrp = entropy(b_data)
    rmdr += value * ( b_count / len(data) )

    return rmdr

def predict(data):
    t = 0
    f = 0
    for i in range(0, len(data)):
        if int(data[i][ len(data[i])-1 ]) == 1:
            t += 1
        else:
            f += 1
    if t > f:
        return 1
    else:
        return 0

def build_tree(node_num, data, prt_sltd_key):
    total_value = entropy(data)
    
    if (total_value < 0.02):
        tree[node_num]['type'] = 2
        tree[node_num]['predict'] = predict(data)
        return
    else:
        info_gain = -9999999
        sltd_key = -1
        sltd_value = -1
        for key in feature_dict.keys():
            if key in ctgl_features:
                temp_info_gain = total_value - remainder(key, data)
                if (temp_info_gain > info_gain) and key not in prt_sltd_key:
                    info_gain = temp_info_gain
                    sltd_key = key
            else:
                values = []
                min = int(feature_dict[key][0])
                max = int(feature_dict[key][1])
                values.append( (min*3 + max*1)/4 )
                values.append( (min*2 + max*2)/4 )
                values.append( (min*1 + max*3)/4 )
                for v in values:
                    temp_info_gain = total_value - remainder_cts(key, v, data)
                    if (temp_info_gain > info_gain) and key not in prt_sltd_key:
                        info_gain = temp_info_gain
                        sltd_key = key
                        sltd_value = v
                        
        if sltd_key == -1:
            tree[node_num]['type'] = 2
            tree[node_num]['predict'] = predict(data)
            return
                    
        tree[node_num]['sltd_key'] = sltd_key
        prt_sltd_key.append(sltd_key)
        
        
        if sltd_key in ctgl_features:
            for value in feature_dict[sltd_key]:
                chd_node_num = len(tree)
                chd_data = []

                for row in data:
                    if row[sltd_key] == value:
                        chd_data.append(row)
                        
                tree[node_num][value] = chd_node_num
                node = {'type': 1}
                tree.append(node)
                build_tree(chd_node_num, chd_data, prt_sltd_key)
        else:
            tree[node_num]['value'] = sltd_value
            
            s_chd_data = []
            b_chd_data = []
            for row in data:
                if int(row[sltd_key]) < sltd_value:
                    s_chd_data.append(row)
                else:
                    b_chd_data.append(row)
            
            chd_node_num = len(tree)
            tree[node_num][1] = chd_node_num
            node = {'type': 1}
            tree.append(node)
            build_tree(chd_node_num, s_chd_data, prt_sltd_key)
            
            chd_node_num = len(tree)
            tree[node_num][0] = chd_node_num
            node = {'type': 1}
            tree.append(node)
            build_tree(chd_node_num, b_chd_data, prt_sltd_key)
        
        prt_sltd_key.remove(sltd_key)
    return
        

#### Main ####
with open('y_train.csv', newline='') as trainfile:
    yrows = list(csv.reader(trainfile))   
    del yrows[0]

with open('X_train.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    #attr = rows[0]
    for i in range (1, len(rows[0])):
        if i in ctgl_features:
            feature_dict[i] = []
        else:
            feature_dict[i] = [99999999, 0]
    del rows[0]
    
for i in range (0, len(rows)):
    rows[i].append(yrows[i][1])
random.shuffle(rows)
    
data_count = len(rows)

tp = [0, 0, 0]
tn = [0, 0, 0]
fp = [0, 0, 0]
fn = [0, 0, 0]

test_range_e = 0
k_ford_k = 3
for k_ford_i in range(k_ford_k):
    print('k', k_ford_i)
    test_range_s = test_range_e
    test_range_e = test_range_s + int(data_count / k_ford_k)
    test_range_e = min(test_range_e, data_count)

    data = []
    for r in range (0, data_count):
        if r in range (test_range_s, test_range_e):
            continue
        row = rows[r]
        for i in range (1, len(row)-1):
    #         if row[i] == ' ?':
    #             continue
            if i in ctgl_features:
                if row[i] not in feature_dict[ i ]:
                    feature_dict[i].append(row[i])
            else:
                if int(row[i]) < feature_dict[i][0]:
                    feature_dict[i][0] = int(row[i])
                elif int(row[i]) > feature_dict[i][1]:
                    feature_dict[i][1] = int(row[i])

        data.append(row)
    
    tree = []
    print('Building tree')    

    node = {'type': 0}
    tree.append(node)
    build_tree(0, data, [])    


    print('Done')

    for r in range(test_range_s, test_range_e):
        row = rows[r]
        node = 0

        while tree[node]['type'] != 2:
            sltd_key = tree[node]['sltd_key']
            if sltd_key in ctgl_features:
                node = tree[node][ row[sltd_key] ]
            else:
                value = tree[node]['value']
                node = tree[node][ int(int(row[sltd_key])<value) ]

        if tree[node]['predict'] == int(row[len(row)-1]):
            if tree[node]['predict'] == 1:
                tp[k_ford_i] += 1
            else:
                tn[k_ford_i] += 1
        else:
            if tree[node]['predict'] == 1:
                fp[k_ford_i] += 1
            else:
                fn[k_ford_i] += 1


print("--------------------------------------------------")
print("K-fold cross-validation")
print("--------------------------------------------------")        
print("Confusion matrix")
print("                     Predict positive      Predict negative")
print("Actual positive     ", statistics.mean(tp), "                 ", statistics.mean(fn))
print("Actual negative     ", statistics.mean(fp), "                 ", statistics.mean(tn))
print("--------------------------------------------------")
print("Accuracy            ", (statistics.mean(tp)+statistics.mean(tn))/(statistics.mean(tp)+statistics.mean(tn)+statistics.mean(fp)+statistics.mean(fn)))
print("Sensitivity (True)  ", (statistics.mean(tp)) / (statistics.mean(tp)+statistics.mean(fn)))
print("Sensitivity (False) ", (statistics.mean(tn)) / (statistics.mean(fp)+statistics.mean(tn)))
print("Precision (True)    ", (statistics.mean(tp)) / (statistics.mean(tp)+statistics.mean(fp)))
print("Precision (False)   ", (statistics.mean(tn)) / (statistics.mean(tn)+statistics.mean(fn)))