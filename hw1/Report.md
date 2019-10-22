2019 Introduction to Machine Learning Program
Assignment #1 - Naïve Bayes
===
:::info
CS10 游騰德 0616026
:::

Objective
===
1. Data Input
    1. Mushroom
       ```python=
       with open('Downloads/agaricus-lepiota.data', newline='') as csvfile:
           rows = list(csv.reader(csvfile))
       ```
    2. Iris
        ```python=
        with open('bezdekIris.data', newline='') as csvfile:
            rows = list(csv.reader(csvfile))
        ```
2. Data Visualization
    :::warning
    Please refer to [here](https://drive.google.com/open?id=1lbkhYAlIu2dbEQnpmTXoh10k97fQlKIx) to view all data visualization
    :::
    1. Mushroom
        ```python=
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
        ```
    2. Iris
        ```python=
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
                    plt.show()
        ```
        Text Output of this python code:
        ```
        Average of sepal length|setosa 5.006
        Standard deviation of sepal length|setosa 0.3524896872134513
        Average of sepal length|versicolor 5.936
        Standard deviation of sepal length|versicolor 0.5161711470638634
        Average of sepal length|virginica 6.588
        Standard deviation of sepal length|virginica 0.6358795932744321
        Average of sepal length 5.843333333333334
        Standard deviation of sepal length 0.8280661279778629
        Average of sepal width|setosa 3.428
        Standard deviation of sepal width|setosa 0.37906436909628866
        Average of sepal width|versicolor 2.77
        Standard deviation of sepal width|versicolor 0.3137983233784114
        Average of sepal width|virginica 2.974
        Standard deviation of sepal width|virginica 0.32249663817263746
        Average of sepal width 3.0573333333333332
        Standard deviation of sepal width 0.4358662849366982
        Average of petal length|setosa 1.462
        Standard deviation of petal length|setosa 0.17366399648018407
        Average of petal length|versicolor 4.26
        Standard deviation of petal length|versicolor 0.46991097723995795
        Average of petal length|virginica 5.552
        Standard deviation of petal length|virginica 0.5518946956639834
        Average of petal length 3.758
        Standard deviation of petal length 1.7652982332594664
        Average of petal width|setosa 0.246
        Standard deviation of petal width|setosa 0.10538558938004565
        Average of petal width|versicolor 1.326
        Standard deviation of petal width|versicolor 0.19775268000454405
        Average of petal width|virginica 2.026
        Standard deviation of petal width|virginica 0.27465005563666733
        Average of petal width 1.1993333333333334
        Standard deviation of petal width 0.7622376689603465
        ```
3. Data processing
    - Drop features with any missing value
        1. Mushroom
            ```python=
            for r in range (len(rows)-1, -1, -1):
            row = rows[r]
            if row[11] is '?':
                del rows[r]
            ```
        2. Iris
            ```python=
            for r in range (len(rows)-1, -1, -1):
            row = rows[r]
            if len(row) is 0:
                del rows[r]
            ```
    - Shuffle
        ```python=
        random.shuffle(rows)
        ```
        
4. Model Construction
    1. Mushroom
        ```python=
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
        ```
    2. Iris
        ```python=
        p_setosa += math.log( GD(float(row[0]), statistics.mean(sl[0]), statistics.stdev(sl[0])) )
        p_setosa += math.log( GD(float(row[1]), statistics.mean(sw[0]), statistics.stdev(sw[0])) )
        p_setosa += math.log( GD(float(row[2]), statistics.mean(pl[0]), statistics.stdev(pl[0])) )
        p_setosa += math.log( GD(float(row[3]), statistics.mean(pw[0]), statistics.stdev(pw[0])) )
        p_setosa += math.log( setosa / (setosa+versicolor+virginica) )
        
        p_versicolor += math.log( GD(float(row[0]), statistics.mean(sl[1]), statistics.stdev(sl[1])) )
        p_versicolor += math.log( GD(float(row[1]), statistics.mean(sw[1]), statistics.stdev(sw[1])) )
        p_versicolor += math.log( GD(float(row[2]), statistics.mean(pl[1]), statistics.stdev(pl[1])) )
        p_versicolor += math.log( GD(float(row[3]), statistics.mean(pw[1]), statistics.stdev(pw[1])) )
        p_versicolor += math.log( versicolor / (setosa+versicolor+virginica) )
        
        p_virginica += math.log( GD(float(row[0]), statistics.mean(sl[2]), statistics.stdev(sl[2])) )
        p_virginica += math.log( GD(float(row[1]), statistics.mean(sw[2]), statistics.stdev(sw[2])) )
        p_virginica += math.log( GD(float(row[2]), statistics.mean(pl[2]), statistics.stdev(pl[2])) )
        p_virginica += math.log( GD(float(row[3]), statistics.mean(pw[2]), statistics.stdev(pw[2])) )
        p_virginica += math.log( virginica / (setosa+versicolor+virginica) )
        ```
        
5. Train-Test-Split
    - Holdout validation
    - K-fold cross-validation
7. Results
    - Holdout validation
        1. Mushroom Result
            ```
            Holdout Validation
            --------------------------------------------------
            Confusion matrix
                             Predict Edible  Predict Poisonous
            Actual Edible     1043            9
            Actual Poisonous  222            419
            --------------------------------------------------
            Accuracy               0.863555818074424
            Precision (Edible)     0.9914448669201521
            Precision (Poisonous)  0.6536661466458659
            Recall (Edible)        0.824505928853755
            Recall (Poisonous)     0.9789719626168224
            --------------------------------------------------
            data_count               5644
            train_data_count         3951
            test_data_count          1693
            Laplace k                3
            ```
        2. Iris Result
            ```
            Holdout validation
            -----------------------------------------------------------------------
            Confusion matrix
                              Predict setosa  Predict versicolor  Predict virginica
            Actual setosa      13              0                   0
            Actual versicolor  0              15                   0
            Actual virginica   0              1                   16
            -----------------------------------------------------------------------
            Accuracy                0.9777777777777777
            Precision (setosa)      1.0
            Precision (versicolor)  1.0
            Precision (virginica)   0.9411764705882353
            Recall (setosa)         0.9285714285714286
            Recall (versicolor)     0.9375
            Recall (virginica)      1.0
            -----------------------------------------------------------------------
            data_count         150
            train_data_count   105
            test_data_count    45
            ```
    - K-fold cross-validation
        1. Mushroom Result
            ```
            K-fold cross-validation
            --------------------------------------------------
            Confusion matrix
                             Predict Edible               Predict Poisonous
            Actual Edible     1156.6666666666667                6
            Actual Poisonous  251.33333333333334                467
            --------------------------------------------------
            Accuracy               0.8631933368775474
            Precision (Edible)     0.9948394495412844
            Precision (Poisonous)  0.6501160092807424
            Recall (Edible)        0.8214962121212122
            Recall (Poisonous)     0.9873150105708245
            --------------------------------------------------
            data_count              5644
            train_data_count (avg)  3763
            test_data_count (avg)   1881
            Laplace k               3
            ```
        2. Iris Result
            ```
            K-fold cross-validation
            -----------------------------------------------------------------------
            Confusion matrix
                              Predict setosa      Predict versicolor      Predict virginica
            Actual setosa      16.666666666666668                  0                       0
            Actual versicolor  0                  15.333333333333334                       1.3333333333333333
            Actual virginica   0                  1.3333333333333333                       15.333333333333334
            -----------------------------------------------------------------------
            Accuracy                0.9466666666666665
            Precision (setosa)      1.0
            Precision (versicolor)  0.9199999999999999
            Precision (virginica)   0.9199999999999999
            Recall (setosa)         0.8620689655172415
            Recall (versicolor)     0.9199999999999999
            Recall (virginica)      0.9199999999999999
            -----------------------------------------------------------------------
            data_count              150
            train_data_count (avg)  100
            test_data_count (avg)   50
            ```
            
7. Comparison & Conclusion
    1. Mushroom
        - With Laplace Smoothing
            ```
            K-fold cross-validation
            --------------------------------------------------
            Confusion matrix
                             Predict Edible               Predict Poisonous
            Actual Edible     1156.6666666666667                6
            Actual Poisonous  251.33333333333334                467
            --------------------------------------------------
            Accuracy               0.8631933368775474
            Precision (Edible)     0.9948394495412844
            Precision (Poisonous)  0.6501160092807424
            Recall (Edible)        0.8214962121212122
            Recall (Poisonous)     0.9873150105708245
            --------------------------------------------------
            data_count              5644
            train_data_count (avg)  3763
            test_data_count (avg)   1881
            Laplace k               3
            ```
        - Without Laplace Smoothing
            ```
            Holdout Validation
            --------------------------------------------------
            Confusion matrix
                             Predict Edible  Predict Poisonous
            Actual Edible     984            44
            Actual Poisonous  302            363
            --------------------------------------------------
            Accuracy               0.7956290608387477
            Precision (Edible)     0.9571984435797666
            Precision (Poisonous)  0.5458646616541354
            Recall (Edible)        0.7651632970451011
            Recall (Poisonous)     0.8918918918918919
            --------------------------------------------------
            data_count               5644
            train_data_count         3951
            test_data_count          1693
            Laplace k                0
            ```
        - Concluson: Laplace smoothing increase the accuracy effciently in the dataset case.

Codes
===
Mushroom dataset with Holdout validation
--
```python=
#Holdout validation
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
    train_data_count = 0
    test_data_count = 0
    TE = 0
    FE = 0
    FP = 0
    TP = 0
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if row[11] is '?':
            del rows[r]
    data_count = len(rows)
    
    random.shuffle(rows)
    test_range_s = 0
    test_range_e = test_range_s + int(data_count * 0.3)
    
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
        train_data_count+=1
    
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
                TE+=1
            elif row[0] is 'p':
                FP+=1
        else:
            # Predict Poisonous
            if row[0] is 'e':
                FE+=1
            elif row[0] is 'p':
                TP+=1
        
        test_data_count+=1
    
# Results
print("Holdout Validation")
print("--------------------------------------------------")
print("Confusion matrix")
print("                 Predict Edible  Predict Poisonous")
print("Actual Edible    ", TE, "          ", FE)
print("Actual Poisonous ", FP, "          ", TP)
print("--------------------------------------------------")
print("Accuracy              ", (TE+TP) / (TE+TP+FE+FP))
print("Precision (Edible)    ", (TE) / (TE+FE))
print("Precision (Poisonous) ", (TP) / (TP+FP))
print("Recall (Edible)       ", (TE) / (TE+FP))
print("Recall (Poisonous)    ", (TP) / (TP+FE))
print("--------------------------------------------------")
print("data_count              ", data_count)
print("train_data_count        ", train_data_count)
print("test_data_count         ", test_data_count)
print("Laplace k               ", k)

###
#                   |   Predict  |    Predict
#                   |   Edible   |   Poisonous
#-------------------+------------+-------------
#  Actual Edible    |     TE     |     FE
#-------------------+------------+-------------
#  Actual Poisonous |     FP     |     TP
#
###
        
```

Mushroom dataset with K-fold cross-validation
--
```python=
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
        
```

Iris dataset with Holdout validation
--
```python=
#Holdout validation
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

    sl = [ [], [], [] ]
    sw = [ [], [], [] ]
    pl = [ [], [], [] ]
    pw = [ [], [], [] ]
    data_count = 0
    train_data_count = 0
    test_data_count = 0
    setosa = 0
    versicolor = 0
    virginica = 0
    TS = 0
    FSC = 0
    FSG = 0
    TG = 0
    FGS = 0
    FGC = 0
    TC = 0
    FCS = 0
    FCG = 0
    
    for r in range (len(rows)-1, -1, -1):
        row = rows[r]
        if len(row) is 0:
            del rows[r]
    data_count = len(rows)
    
    random.shuffle(rows)
    test_range_s = 0
    test_range_e = test_range_s + int(data_count * 0.3)
    
    # Train
    for r in range (0, data_count):
        if r in range (test_range_s, test_range_e):
            continue
        
        row = rows[r]
        
        iris = 0
        if row[4] == 'Iris-setosa':
            iris = 0
            setosa+=1
        elif row[4] == 'Iris-versicolor':
            iris = 1
            versicolor+=1
        elif row[4] == 'Iris-virginica':
            iris = 2
            virginica+=1
        
        sl[iris].append(float(row[0]))
        sw[iris].append(float(row[1]))
        pl[iris].append(float(row[2]))
        pw[iris].append(float(row[3]))
        
        train_data_count+=1
    
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
        p_setosa += math.log( setosa / (setosa+versicolor+virginica) )
        
        p_versicolor += math.log( GD(float(row[0]), statistics.mean(sl[1]), statistics.stdev(sl[1])) )
        p_versicolor += math.log( GD(float(row[1]), statistics.mean(sw[1]), statistics.stdev(sw[1])) )
        p_versicolor += math.log( GD(float(row[2]), statistics.mean(pl[1]), statistics.stdev(pl[1])) )
        p_versicolor += math.log( GD(float(row[3]), statistics.mean(pw[1]), statistics.stdev(pw[1])) )
        p_versicolor += math.log( versicolor / (setosa+versicolor+virginica) )
        
        p_virginica += math.log( GD(float(row[0]), statistics.mean(sl[2]), statistics.stdev(sl[2])) )
        p_virginica += math.log( GD(float(row[1]), statistics.mean(sw[2]), statistics.stdev(sw[2])) )
        p_virginica += math.log( GD(float(row[2]), statistics.mean(pl[2]), statistics.stdev(pl[2])) )
        p_virginica += math.log( GD(float(row[3]), statistics.mean(pw[2]), statistics.stdev(pw[2])) )
        p_virginica += math.log( virginica / (setosa+versicolor+virginica) )
        
        max_p = max(p_setosa, p_versicolor, p_virginica)
        
        if max_p == p_setosa:
            # Predict setosa
            if row[4] == 'Iris-setosa':
                TS+=1
            elif row[4] == 'Iris-versicolor':
                FSC+=1
            elif row[4] == 'Iris-virginica':
                FSG+=1
        elif max_p == p_versicolor:
            # Predict versicolor
            if row[4] == 'Iris-setosa':
                FCS+=1
            elif row[4] == 'Iris-versicolor':
                TC+=1
            elif row[4] == 'Iris-virginica':
                FCG+=1
        elif max_p == p_virginica:
            # Predict virginica
            if row[4] == 'Iris-setosa':
                FGS+=1
            elif row[4] == 'Iris-versicolor':
                FGC+=1
            elif row[4] == 'Iris-virginica':
                TG+=1
                
        test_data_count+=1
    
# Results
print("Holdout validation")
print("-----------------------------------------------------------------------")
print("Confusion matrix")
print("                  Predict setosa  Predict versicolor  Predict virginica")
print("Actual setosa     ", TS, "            ", FSC, "                 ", FSG)
print("Actual versicolor ", FCS, "            ", TC, "                 ", FCG)
print("Actual virginica  ", FGS, "            ", FGC, "                 ", TG)
print("-----------------------------------------------------------------------")
print("Accuracy               ", (TS+TC+TG) / (TS+TC+TG+FSC+FSG+FCS+FCG+FGS+FGC))
print("Precision (setosa)     ", (TS) / (TS+FSC+FSG))
print("Precision (versicolor) ", (TC) / (TC+FCS+FCG))
print("Precision (virginica)  ", (TG) / (TG+FGS+FGC))
print("Recall (setosa)        ", (TS) / (TS+FCS+FCG+FGS+FGC))
print("Recall (versicolor)    ", (TC) / (TC+FSC+FSG+FGS+FGC))
print("Recall (virginica)     ", (TG) / (TG+FSC+FSG+FCS+FCG))
print("-----------------------------------------------------------------------")
print("data_count        ", data_count)
print("train_data_count  ", train_data_count)
print("test_data_count   ", test_data_count)

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
```

Iris dataset with K-fold cross-validation
--
```python=
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
```

###### tags: `NCTU` `ML` `Naïve Bayes`
