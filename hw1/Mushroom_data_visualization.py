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

with open('agaricus-lepiota.data', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    feature_name = ["label", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    feature_list_all = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    feature_list_e = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    feature_list_p = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if row[11] is '?':
            del rows[r]
    data_count = len(rows)
    
    for r in range (0, data_count):
        row = rows[r]
        
        label = row[0]
        if label in feature_list_all[0]:
            feature_list_all[0][label]+=1
        else:
            feature_list_all[0][label] = 1
        
        for i in range (1, 23):
            feature = row[i]
            if feature in feature_list_all[i]:
                feature_list_all[i][feature]+=1
            else:
                feature_list_all[i][feature] = 1
            
            if label is 'e':
                if feature in feature_list_e[i]:
                    feature_list_e[i][feature]+=1
                else:
                    feature_list_e[i][feature] = 1
            elif label is 'p':
                if feature in feature_list_p[i]:
                    feature_list_p[i][feature]+=1
                else:
                    feature_list_p[i][feature] = 1
    
    for f in range(24):
        if f is 0:
            keys = feature_list_all[f].keys()
            vals = feature_list_all[f].values()
            plt.bar(keys, np.divide(list(vals), sum(vals)), label=feature_name[f]+" distribution")
            plt.ylim(0,1)
            plt.ylabel ('Percentage')
            plt.xlabel (feature_name[f])
            plt.xticks(list(keys))
            plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
            if savefig:
                plt.savefig("Visualization/Mushroom/"+feature_name[f]+"_distribution.png")
            plt.show()
        else:
            keys = feature_list_all[f].keys()
            vals = feature_list_all[f].values()
            plt.bar(keys, np.divide(list(vals), sum(vals)), label=feature_name[f]+" distribution")
            plt.ylim(0,1)
            plt.ylabel ('Percentage')
            plt.xlabel (feature_name[f])
            plt.xticks(list(keys))
            plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
            if savefig:
                plt.savefig("Visualization/Mushroom/"+feature_name[f]+"_distribution.png")
            plt.show()
            
            keys = feature_list_e[f].keys()
            vals = feature_list_e[f].values()
            plt.bar(keys, np.divide(list(vals), sum(vals)), label=feature_name[f]+"|edible distribution")
            plt.ylim(0,1)
            plt.ylabel ('Percentage')
            plt.xlabel (feature_name[f])
            plt.xticks(list(keys))
            plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
            if savefig:
                plt.savefig("Visualization/Mushroom/"+feature_name[f]+"|edible_distribution.png")
            plt.show()
            
            keys = feature_list_p[f].keys()
            vals = feature_list_p[f].values()
            plt.bar(keys, np.divide(list(vals), sum(vals)), label=feature_name[f]+"|poisionous distribution")
            plt.ylim(0,1)
            plt.ylabel ('Percentage')
            plt.xlabel (feature_name[f])
            plt.xticks(list(keys))
            plt.legend (bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.)
            if savefig:
                plt.savefig("Visualization/Mushroom/"+feature_name[f]+"|poisionous_distribution.png")
            plt.show()