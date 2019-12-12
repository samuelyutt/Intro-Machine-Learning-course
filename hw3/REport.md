2019 Introduction to Machine Learning Program
Assignment #2 - Decision tree and Random forest
===
:::info
CS10 游騰德 0616026
:::

Objective
===
1. Data Input
    ```python=
    with open('y_train.csv', newline='') as trainfile:
        yrows = list(csv.reader(trainfile))   
        del yrows[0]

    with open('X_train.csv', newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        # Construct the dictionary of features
    ```

2. Data Preprocessing
    1. Transform data format
        ```python=8
        # Contd.
            for i in range (1, len(rows[0])):
                if i in ctgl_features:
                    feature_dict[i] = []
                else:
                    feature_dict[i] = [99999999, 0]
            del rows[0]

        for i in range (0, len(rows)):
            rows[i].append(yrows[i][1])
        ```
    2. Suffle the data
        ```python= 
        random.shuffle(rows)
        ```
3. Model Constuction
    1. Categorical and continuous data
        - Categorical data
            ```python= 
            if key in ctgl_features:
                temp_info_gain = total_value - remainder(key, data)
                if (temp_info_gain > info_gain) and key not in prt_sltd_key:
                    info_gain = temp_info_gain
                    sltd_key = key
            ```
        - Continuous data
            ```python=6
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
            ```
    2. Entropy and Gini index
        - Entropy
            ```python=
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
            ```
        - Gini index
            ```python=
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
            ```
    3. Decision Tree model
        ```python= 
        def build_tree(node_num, data, prt_sltd_key):
            if entropy or gini_index < 0.02:
                # Make this node a leaf node
                # Predict
                return
            else:
                for key in feature_dict.keys():
                    if key in ctgl_features:
                        # Calculate info-gain for categorical feature
                    else:
                        # Calculate info-gain for continuous feature by different values

                # Select the best feature to split data   

                if sltd_key in ctgl_features:
                    # Append nodes to the tree by different features among the selected feature
                    for value in feature_dict[sltd_key]:
                        # Append child node
                else:
                    # Append two nodes to the tree by the condition among the selected feature 
                    # Append child node (a < value)
                    # Append child node (a >= value)
            return
        ```
    4. Random Forest model
        - Data selection: Randomly select features
            ```python= 
            for tree_i in range (0, tree_count):
                # Random choose features for each tree
                features = []
                while len(features) < tree_feature_count:
                    new_feature = random.randint(1, 14)
                    if new_feature not in features:
                        features.append(new_feature)
                tree_feature.append(features)

                # Build a tree
                tree = [] 
                node = {'type': 0}
                tree.append(node)
                build_tree(0, data, [], tree_i)    

                # Append the tree to the forest
                forest.append(tree)

            ```
        - Number of trees: As you may see in the above codes, the procedure create `tree_count` trees and choose `tree_feature_count` features for them. Here I choose to set
            ```python= 
            tree_count = 10
            tree_feature_count = 8
            ```
        - Difference between K-ford cross-validation and Random Forest
            - K-ford cross-validation chooses 1/k data to be validation data and left the other to be training data. After one train finished, it will choose another 1/k data and so on. K-ford cross-validation generates the result after k times of training and validating by averaging the results from each precedure.
            - Random Forest generates a forest by building an amount of Decision Trees. When predicting data, every Decision Tree votes for a prediction. The model predict by choosing the maximum vote.

4. Validation
    1. Holdout validation
        ```python= 
        random.shuffle(rows)
        test_range_s = 0
        test_range_e = test_range_s + int(data_count * 0.3)

        for r in range (0, data_count):
            if r not in range (test_range_s, test_range_e):
            # Train

        for r in range (test_range_s, test_range_e):
            # Validate
        ```
    2. K-fold cross-validation
        ```python= 
        test_range_e = 0
        k_ford_k = 3
        for k_ford_i in range(k_ford_k):          
            test_range_s = test_range_e
            test_range_e = test_range_s + int(data_count / k_ford_k)
            test_range_e = min(test_range_e, data_count)

            for r in range (0, data_count):
                if r not in range (test_range_s, test_range_e):
                # Train

            for r in range (test_range_s, test_range_e):
                # Validate
        ```

5. Results
    1. Decision tree
        - Holdout validation
            ```
            --------------------------------------------------
            Holdout Validation
            --------------------------------------------------
            Confusion matrix
                                 Predict positive      Predict negative
            Actual positive      839                   859
            Actual negative      494                   4645
            --------------------------------------------------
            Accuracy             0.8021061869240895
            Sensitivity (True)   0.49411071849234395
            Sensitivity (False)  0.9038723487059739
            Precision (True)     0.6294073518379595
            Precision (False)    0.8439316860465116
            ```
        - K-fold cross-validation
            ```
            --------------------------------------------------
            K-fold cross-validation
            --------------------------------------------------
            Confusion matrix
                                 Predict positive      Predict negative
            Actual positive      912.6666666666666     929.6666666666666
            Actual negative      539.6666666666666     5215
            --------------------------------------------------
            Accuracy             0.8065903207406432
            Sensitivity (True)   0.4953862855075086
            Sensitivity (False)  0.9062210379981464
            Precision (True)     0.6284140463621758
            Precision (False)    0.84870348269502
            ```
        - Show the prediction and reasoning of 10 samples in the validation set.
            ```cpp=-
            Sample 1 : ['15240', '55', ' Self-emp-not-inc', '161334', ' HS-grad', '9', ' Widowed', ' Other-service', ' Not-in-family', ' White', ' Female', '0', '0', '25', ' Nicaragua', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 957
            |
            v
            Current in node 957
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 958, 0: 1484}
            Action: Go to child node 1484
            |
            v
            Current in node 1484
            {'type': 1, 'sltd_key': 8, ' Husband': 1485, ' Own-child': 2866, ' Unmarried': 2867, ' Not-in-family': 2926, ' Wife': 3231, ' Other-relative': 3639}
            Action: Go to child node 2926
            |
            v
            Current in node 2926
            {'type': 1, 'sltd_key': 4, ' Some-college': 2927, ' HS-grad': 2966, ' Doctorate': 3122, ' Masters': 3139, ' Bachelors': 3162, ' Assoc-acdm': 3220, ' Assoc-voc': 3221, ' 10th': 3222, ' 5th-6th': 3223, ' 12th': 3224, ' 11th': 3225, ' 1st-4th': 3226, ' 7th-8th': 3227, ' 9th': 3228, ' Prof-school': 3229, ' Preschool': 3230}
            Action: Go to child node 2966
            |
            v
            Current in node 2966
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 2967, ' Protective-serv': 2968, ' Transport-moving': 2969, ' Machine-op-inspct': 2970, ' Sales': 2971, ' Exec-managerial': 2972, ' Prof-specialty': 3043, ' Adm-clerical': 3044, ' Farming-fishing': 3045, ' Tech-support': 3046, ' ?': 3047, ' Priv-house-serv': 3118, ' Other-service': 3119, ' Handlers-cleaners': 3120, ' Armed-Forces': 3121}
            Action: Go to child node 3119
            |
            v
            Current in node 3119
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 2 : ['16493', '48', ' Self-emp-inc', '481987', ' 10th', '6', ' Married-civ-spouse', ' Exec-managerial', ' Husband', ' White', ' Male', '0', '0', '40', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 14574
            |
            v
            Current in node 14574
            {'type': 1, 'sltd_key': 8, ' Husband': 14575, ' Own-child': 26602, ' Unmarried': 26958, ' Not-in-family': 29115, ' Wife': 33500, ' Other-relative': 35284}
            Action: Go to child node 14575
            |
            v
            Current in node 14575
            {'type': 1, 'sltd_key': 4, ' Some-college': 14576, ' HS-grad': 16956, ' Doctorate': 19532, ' Masters': 19820, ' Bachelors': 20988, ' Assoc-acdm': 22650, ' Assoc-voc': 23710, ' 10th': 24883, ' 5th-6th': 25378, ' 12th': 25379, ' 11th': 25553, ' 1st-4th': 26000, ' 7th-8th': 26001, ' 9th': 26002, ' Prof-school': 26253, ' Preschool': 26601}
            Action: Go to child node 24883
            |
            v
            Current in node 24883
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 24884, ' Protective-serv': 25060, ' Transport-moving': 25061, ' Machine-op-inspct': 25145, ' Sales': 25216, ' Exec-managerial': 25219, ' Prof-specialty': 25290, ' Adm-clerical': 25291, ' Farming-fishing': 25292, ' Tech-support': 25293, ' ?': 25294, ' Priv-house-serv': 25365, ' Other-service': 25366, ' Handlers-cleaners': 25367, ' Armed-Forces': 25377}
            Action: Go to child node 25219
            |
            v
            Current in node 25219
            {'type': 1, 'sltd_key': 2, ' Private': 25220, ' Local-gov': 25282, ' Federal-gov': 25283, ' Self-emp-not-inc': 25284, ' ?': 25285, ' Self-emp-inc': 25286, ' State-gov': 25287, ' Without-pay': 25288, ' Never-worked': 25289}
            Action: Go to child node 25286
            |
            v
            Current in node 25286
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 3 : ['3405', '52', ' State-gov', '104280', ' Some-college', '10', ' Married-civ-spouse', ' Tech-support', ' Husband', ' White', ' Male', '0', '0', '50', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 14574
            |
            v
            Current in node 14574
            {'type': 1, 'sltd_key': 8, ' Husband': 14575, ' Own-child': 26602, ' Unmarried': 26958, ' Not-in-family': 29115, ' Wife': 33500, ' Other-relative': 35284}
            Action: Go to child node 14575
            |
            v
            Current in node 14575
            {'type': 1, 'sltd_key': 4, ' Some-college': 14576, ' HS-grad': 16956, ' Doctorate': 19532, ' Masters': 19820, ' Bachelors': 20988, ' Assoc-acdm': 22650, ' Assoc-voc': 23710, ' 10th': 24883, ' 5th-6th': 25378, ' 12th': 25379, ' 11th': 25553, ' 1st-4th': 26000, ' 7th-8th': 26001, ' 9th': 26002, ' Prof-school': 26253, ' Preschool': 26601}
            Action: Go to child node 14576
            |
            v
            Current in node 14576
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 14577, ' Protective-serv': 14816, ' Transport-moving': 15009, ' Machine-op-inspct': 15146, ' Sales': 15219, ' Exec-managerial': 15512, ' Prof-specialty': 15871, ' Adm-clerical': 16028, ' Farming-fishing': 16151, ' Tech-support': 16385, ' ?': 16456, ' Priv-house-serv': 16590, ' Other-service': 16591, ' Handlers-cleaners': 16784, ' Armed-Forces': 16955}
            Action: Go to child node 16385
            |
            v
            Current in node 16385
            {'type': 1, 'sltd_key': 2, ' Private': 16386, ' Local-gov': 16448, ' Federal-gov': 16449, ' Self-emp-not-inc': 16450, ' ?': 16451, ' Self-emp-inc': 16452, ' State-gov': 16453, ' Without-pay': 16454, ' Never-worked': 16455}
            Action: Go to child node 16453
            |
            v
            Current in node 16453
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 4 : ['11697', '33', ' Self-emp-not-inc', '272359', ' Bachelors', '13', ' Married-civ-spouse', ' Sales', ' Husband', ' White', ' Male', '7298', '0', '80', ' United-States', '1']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 3679
            |
            v
            Current in node 3679
            {'type': 1, 'sltd_key': 8, ' Husband': 3680, ' Own-child': 9198, ' Unmarried': 9726, ' Not-in-family': 9940, ' Wife': 12794, ' Other-relative': 14446}
            Action: Go to child node 3680
            |
            v
            Current in node 3680
            {'type': 1, 'sltd_key': 4, ' Some-college': 3681, ' HS-grad': 5092, ' Doctorate': 6843, ' Masters': 6964, ' Bachelors': 7303, ' Assoc-acdm': 7919, ' Assoc-voc': 8335, ' 10th': 8993, ' 5th-6th': 9018, ' 12th': 9019, ' 11th': 9022, ' 1st-4th': 9108, ' 7th-8th': 9109, ' 9th': 9110, ' Prof-school': 9111, ' Preschool': 9197}
            Action: Go to child node 7303
            |
            v
            Current in node 7303
            {'type': 1, 'sltd_key': 14, ' United-States': 7304, ' Puerto-Rico': 7801, ' Poland': 7802, ' Vietnam': 7803, ' Cuba': 7804, ' Philippines': 7805, ' Mexico': 7806, ' India': 7807, ' Columbia': 7817, ' Taiwan': 7818, ' ?': 7819, ' Japan': 7871, ' Hungary': 7872, ' Hong': 7873, ' South': 7874, ' El-Salvador': 7875, ' China': 7876, ' Dominican-Republic': 7877, ' Germany': 7878, ' Canada': 7888, ' Iran': 7889, ' Yugoslavia': 7899, ' Italy': 7900, ' Nicaragua': 7901, ' Peru': 7902, ' Ecuador': 7903, ' Portugal': 7904, ' Jamaica': 7905, ' Outlying-US(Guam-USVI-etc)': 7906, ' Haiti': 7907, ' Greece': 7908, ' Guatemala': 7909, ' Trinadad&Tobago': 7910, ' England': 7911, ' Scotland': 7912, ' France': 7913, ' Cambodia': 7914, ' Thailand': 7915, ' Honduras': 7916, ' Ireland': 7917, ' Laos': 7918}
            Action: Go to child node 7304
            |
            v
            Current in node 7304
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 7305, ' Protective-serv': 7355, ' Transport-moving': 7392, ' Machine-op-inspct': 7398, ' Sales': 7410, ' Exec-managerial': 7475, ' Prof-specialty': 7576, ' Adm-clerical': 7677, ' Farming-fishing': 7707, ' Tech-support': 7737, ' ?': 7767, ' Priv-house-serv': 7768, ' Other-service': 7769, ' Handlers-cleaners': 7799, ' Armed-Forces': 7800}
            Action: Go to child node 7410
            |
            v
            Current in node 7410
            {'type': 1, 'sltd_key': 9, ' White': 7411, ' Black': 7471, ' Asian-Pac-Islander': 7472, ' Amer-Indian-Eskimo': 7473, ' Other': 7474}
            Action: Go to child node 7411
            |
            v
            Current in node 7411
            {'type': 1, 'sltd_key': 3, 'value': 385652.75, 1: 7412, 0: 7448}
            Action: Go to child node 7412
            |
            v
            Current in node 7412
            {'type': 1, 'sltd_key': 2, ' Private': 7413, ' Local-gov': 7427, ' Federal-gov': 7428, ' Self-emp-not-inc': 7429, ' ?': 7430, ' Self-emp-inc': 7431, ' State-gov': 7445, ' Without-pay': 7446, ' Never-worked': 7447}
            Action: Go to child node 7429
            |
            v
            Current in node 7429
            {'type': 2, 'predict': 1}
            Action: Predict True


            Sample 5 : ['14827', '23', ' Private', '113601', ' Some-college', '10', ' Never-married', ' Handlers-cleaners', ' Own-child', ' White', ' Male', '0', '0', '30', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 3679
            |
            v
            Current in node 3679
            {'type': 1, 'sltd_key': 8, ' Husband': 3680, ' Own-child': 9198, ' Unmarried': 9726, ' Not-in-family': 9940, ' Wife': 12794, ' Other-relative': 14446}
            Action: Go to child node 9198
            |
            v
            Current in node 9198
            {'type': 1, 'sltd_key': 4, ' Some-college': 9199, ' HS-grad': 9202, ' Doctorate': 9203, ' Masters': 9204, ' Bachelors': 9356, ' Assoc-acdm': 9538, ' Assoc-voc': 9539, ' 10th': 9632, ' 5th-6th': 9633, ' 12th': 9634, ' 11th': 9635, ' 1st-4th': 9636, ' 7th-8th': 9637, ' 9th': 9638, ' Prof-school': 9639, ' Preschool': 9725}
            Action: Go to child node 9199
            |
            v
            Current in node 9199
            {'type': 1, 'sltd_key': 11, 'value': 24999.75, 1: 9200, 0: 9201}
            Action: Go to child node 9200
            |
            v
            Current in node 9200
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 6 : ['19302', '68', ' Self-emp-inc', '136218', ' HS-grad', '9', ' Widowed', ' Other-service', ' Unmarried', ' White', ' Female', '0', '0', '15', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 957
            |
            v
            Current in node 957
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 958, 0: 1484}
            Action: Go to child node 1484
            |
            v
            Current in node 1484
            {'type': 1, 'sltd_key': 8, ' Husband': 1485, ' Own-child': 2866, ' Unmarried': 2867, ' Not-in-family': 2926, ' Wife': 3231, ' Other-relative': 3639}
            Action: Go to child node 2867
            |
            v
            Current in node 2867
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 2868, ' Protective-serv': 2869, ' Transport-moving': 2870, ' Machine-op-inspct': 2871, ' Sales': 2872, ' Exec-managerial': 2873, ' Prof-specialty': 2874, ' Adm-clerical': 2884, ' Farming-fishing': 2919, ' Tech-support': 2920, ' ?': 2921, ' Priv-house-serv': 2922, ' Other-service': 2923, ' Handlers-cleaners': 2924, ' Armed-Forces': 2925}
            Action: Go to child node 2923
            |
            v
            Current in node 2923
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 7 : ['17072', '31', ' State-gov', '75755', ' Doctorate', '16', ' Married-civ-spouse', ' Exec-managerial', ' Husband', ' White', ' Male', '7298', '0', '55', ' United-States', '1']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 3679
            |
            v
            Current in node 3679
            {'type': 1, 'sltd_key': 8, ' Husband': 3680, ' Own-child': 9198, ' Unmarried': 9726, ' Not-in-family': 9940, ' Wife': 12794, ' Other-relative': 14446}
            Action: Go to child node 3680
            |
            v
            Current in node 3680
            {'type': 1, 'sltd_key': 4, ' Some-college': 3681, ' HS-grad': 5092, ' Doctorate': 6843, ' Masters': 6964, ' Bachelors': 7303, ' Assoc-acdm': 7919, ' Assoc-voc': 8335, ' 10th': 8993, ' 5th-6th': 9018, ' 12th': 9019, ' 11th': 9022, ' 1st-4th': 9108, ' 7th-8th': 9109, ' 9th': 9110, ' Prof-school': 9111, ' Preschool': 9197}
            Action: Go to child node 6843
            |
            v
            Current in node 6843
            {'type': 1, 'sltd_key': 14, ' United-States': 6844, ' Puerto-Rico': 6924, ' Poland': 6925, ' Vietnam': 6926, ' Cuba': 6927, ' Philippines': 6928, ' Mexico': 6929, ' India': 6930, ' Columbia': 6931, ' Taiwan': 6932, ' ?': 6933, ' Japan': 6934, ' Hungary': 6935, ' Hong': 6936, ' South': 6937, ' El-Salvador': 6938, ' China': 6939, ' Dominican-Republic': 6940, ' Germany': 6941, ' Canada': 6942, ' Iran': 6943, ' Yugoslavia': 6944, ' Italy': 6945, ' Nicaragua': 6946, ' Peru': 6947, ' Ecuador': 6948, ' Portugal': 6949, ' Jamaica': 6950, ' Outlying-US(Guam-USVI-etc)': 6951, ' Haiti': 6952, ' Greece': 6953, ' Guatemala': 6954, ' Trinadad&Tobago': 6955, ' England': 6956, ' Scotland': 6957, ' France': 6958, ' Cambodia': 6959, ' Thailand': 6960, ' Honduras': 6961, ' Ireland': 6962, ' Laos': 6963}
            Action: Go to child node 6844
            |
            v
            Current in node 6844
            {'type': 1, 'sltd_key': 2, ' Private': 6845, ' Local-gov': 6881, ' Federal-gov': 6882, ' Self-emp-not-inc': 6883, ' ?': 6884, ' Self-emp-inc': 6885, ' State-gov': 6921, ' Without-pay': 6922, ' Never-worked': 6923}
            Action: Go to child node 6921
            |
            v
            Current in node 6921
            {'type': 2, 'predict': 1}
            Action: Predict True


            Sample 8 : ['18716', '38', ' Private', '107125', ' Some-college', '10', ' Never-married', ' Sales', ' Not-in-family', ' White', ' Male', '0', '0', '60', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 14574
            |
            v
            Current in node 14574
            {'type': 1, 'sltd_key': 8, ' Husband': 14575, ' Own-child': 26602, ' Unmarried': 26958, ' Not-in-family': 29115, ' Wife': 33500, ' Other-relative': 35284}
            Action: Go to child node 29115
            |
            v
            Current in node 29115
            {'type': 1, 'sltd_key': 4, ' Some-college': 29116, ' HS-grad': 29908, ' Doctorate': 31027, ' Masters': 31150, ' Bachelors': 31767, ' Assoc-acdm': 32772, ' Assoc-voc': 32928, ' 10th': 33172, ' 5th-6th': 33188, ' 12th': 33189, ' 11th': 33190, ' 1st-4th': 33283, ' 7th-8th': 33284, ' 9th': 33285, ' Prof-school': 33303, ' Preschool': 33499}
            Action: Go to child node 29116
            |
            v
            Current in node 29116
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 29117, ' Protective-serv': 29188, ' Transport-moving': 29196, ' Machine-op-inspct': 29267, ' Sales': 29275, ' Exec-managerial': 29353, ' Prof-specialty': 29459, ' Adm-clerical': 29584, ' Farming-fishing': 29720, ' Tech-support': 29721, ' ?': 29763, ' Priv-house-serv': 29764, ' Other-service': 29765, ' Handlers-cleaners': 29836, ' Armed-Forces': 29907}
            Action: Go to child node 29275
            |
            v
            Current in node 29275
            {'type': 1, 'sltd_key': 2, ' Private': 29276, ' Local-gov': 29338, ' Federal-gov': 29339, ' Self-emp-not-inc': 29340, ' ?': 29341, ' Self-emp-inc': 29342, ' State-gov': 29350, ' Without-pay': 29351, ' Never-worked': 29352}
            Action: Go to child node 29276
            |
            v
            Current in node 29276
            {'type': 1, 'sltd_key': 6, ' Married-civ-spouse': 29277, ' Never-married': 29278, ' Divorced': 29279, ' Widowed': 29280, ' Separated': 29281, ' Married-spouse-absent': 29336, ' Married-AF-spouse': 29337}
            Action: Go to child node 29278
            |
            v
            Current in node 29278
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 9 : ['17394', '22', ' Private', '279802', ' Some-college', '10', ' Never-married', ' Adm-clerical', ' Own-child', ' White', ' Female', '0', '0', '40', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 3679
            |
            v
            Current in node 3679
            {'type': 1, 'sltd_key': 8, ' Husband': 3680, ' Own-child': 9198, ' Unmarried': 9726, ' Not-in-family': 9940, ' Wife': 12794, ' Other-relative': 14446}
            Action: Go to child node 9198
            |
            v
            Current in node 9198
            {'type': 1, 'sltd_key': 4, ' Some-college': 9199, ' HS-grad': 9202, ' Doctorate': 9203, ' Masters': 9204, ' Bachelors': 9356, ' Assoc-acdm': 9538, ' Assoc-voc': 9539, ' 10th': 9632, ' 5th-6th': 9633, ' 12th': 9634, ' 11th': 9635, ' 1st-4th': 9636, ' 7th-8th': 9637, ' 9th': 9638, ' Prof-school': 9639, ' Preschool': 9725}
            Action: Go to child node 9199
            |
            v
            Current in node 9199
            {'type': 1, 'sltd_key': 11, 'value': 24999.75, 1: 9200, 0: 9201}
            Action: Go to child node 9200
            |
            v
            Current in node 9200
            {'type': 2, 'predict': 0}
            Action: Predict False


            Sample 10 : ['2934', '20', ' Private', '353195', ' HS-grad', '9', ' Never-married', ' Craft-repair', ' Not-in-family', ' White', ' Male', '0', '0', '35', ' United-States', '0']
            Current in node 0
            {'type': 0, 'sltd_key': 5, 'value': 4.75, 1: 1, 0: 956}
            Action: Go to child node 956
            |
            v
            Current in node 956
            {'type': 1, 'sltd_key': 13, 'value': 25.5, 1: 957, 0: 3678}
            Action: Go to child node 3678
            |
            v
            Current in node 3678
            {'type': 1, 'sltd_key': 1, 'value': 35.25, 1: 3679, 0: 14574}
            Action: Go to child node 3679
            |
            v
            Current in node 3679
            {'type': 1, 'sltd_key': 8, ' Husband': 3680, ' Own-child': 9198, ' Unmarried': 9726, ' Not-in-family': 9940, ' Wife': 12794, ' Other-relative': 14446}
            Action: Go to child node 9940
            |
            v
            Current in node 9940
            {'type': 1, 'sltd_key': 4, ' Some-college': 9941, ' HS-grad': 10202, ' Doctorate': 10956, ' Masters': 11066, ' Bachelors': 11427, ' Assoc-acdm': 12256, ' Assoc-voc': 12498, ' 10th': 12600, ' 5th-6th': 12601, ' 12th': 12602, ' 11th': 12625, ' 1st-4th': 12626, ' 7th-8th': 12627, ' 9th': 12628, ' Prof-school': 12670, ' Preschool': 12793}
            Action: Go to child node 10202
            |
            v
            Current in node 10202
            {'type': 1, 'sltd_key': 2, ' Private': 10203, ' Local-gov': 10639, ' Federal-gov': 10662, ' Self-emp-not-inc': 10663, ' ?': 10860, ' Self-emp-inc': 10861, ' State-gov': 10938, ' Without-pay': 10954, ' Never-worked': 10955}
            Action: Go to child node 10203
            |
            v
            Current in node 10203
            {'type': 1, 'sltd_key': 7, ' Craft-repair': 10204, ' Protective-serv': 10266, ' Transport-moving': 10267, ' Machine-op-inspct': 10329, ' Sales': 10330, ' Exec-managerial': 10392, ' Prof-specialty': 10454, ' Adm-clerical': 10455, ' Farming-fishing': 10517, ' Tech-support': 10518, ' ?': 10519, ' Priv-house-serv': 10520, ' Other-service': 10521, ' Handlers-cleaners': 10637, ' Armed-Forces': 10638}
            Action: Go to child node 10204
            |
            v
            Current in node 10204
            {'type': 1, 'sltd_key': 6, ' Married-civ-spouse': 10205, ' Never-married': 10206, ' Divorced': 10261, ' Widowed': 10262, ' Separated': 10263, ' Married-spouse-absent': 10264, ' Married-AF-spouse': 10265}
            Action: Go to child node 10206
            |
            v
            Current in node 10206
            {'type': 1, 'sltd_key': 10, ' Male': 10207, ' Female': 10260}
            Action: Go to child node 10207
            |
            v
            Current in node 10207
            {'type': 1, 'sltd_key': 9, ' White': 10208, ' Black': 10256, ' Asian-Pac-Islander': 10257, ' Amer-Indian-Eskimo': 10258, ' Other': 10259}
            Action: Go to child node 10208
            |
            v
            Current in node 10208
            {'type': 1, 'sltd_key': 14, ' United-States': 10209, ' Puerto-Rico': 10216, ' Poland': 10217, ' Vietnam': 10218, ' Cuba': 10219, ' Philippines': 10220, ' Mexico': 10221, ' India': 10222, ' Columbia': 10223, ' Taiwan': 10224, ' ?': 10225, ' Japan': 10226, ' Hungary': 10227, ' Hong': 10228, ' South': 10229, ' El-Salvador': 10230, ' China': 10231, ' Dominican-Republic': 10232, ' Germany': 10233, ' Canada': 10234, ' Iran': 10235, ' Yugoslavia': 10236, ' Italy': 10237, ' Nicaragua': 10238, ' Peru': 10239, ' Ecuador': 10240, ' Portugal': 10241, ' Jamaica': 10242, ' Outlying-US(Guam-USVI-etc)': 10243, ' Haiti': 10244, ' Greece': 10245, ' Guatemala': 10246, ' Trinadad&Tobago': 10247, ' England': 10248, ' Scotland': 10249, ' France': 10250, ' Cambodia': 10251, ' Thailand': 10252, ' Honduras': 10253, ' Ireland': 10254, ' Laos': 10255}
            Action: Go to child node 10209
            |
            v
            Current in node 10209
            {'type': 1, 'sltd_key': 3, 'value': 752003.5, 1: 10210, 0: 10215}
            Action: Go to child node 10210
            |
            v
            Current in node 10210
            {'type': 1, 'sltd_key': 11, 'value': 24999.75, 1: 10211, 0: 10214}
            Action: Go to child node 10211
            |
            v
            Current in node 10211
            {'type': 1, 'sltd_key': 12, 'value': 2178.0, 1: 10212, 0: 10213}
            Action: Go to child node 10212
            |
            v
            Current in node 10212
            {'type': 2, 'predict': 0}
            Action: Predict False


            --------------------------------------------------
            Holdout Validation
            --------------------------------------------------
            Confusion matrix
                                 Predict positive    Predict negative
            Actual positive      2                   0
            Actual negative      0                   8
            --------------------------------------------------
            Accuracy             1.0
            Sensitivity (True)   1.0
            Sensitivity (False)  1.0
            Precision (True)     1.0
            Precision (False)    1.0
            ```
    2. Random Forest
        - Holdout validation
            ```
            --------------------------------------------------
            Holdout Validation
            --------------------------------------------------
            Confusion matrix
                                 Predict positive      Predict negative
            Actual positive      784                   810
            Actual negative      340                   4903
            --------------------------------------------------
            Accuracy             0.8317975720345181
            Sensitivity (True)   0.4918444165621079
            Sensitivity (False)  0.9351516307457562
            Precision (True)     0.697508896797153
            Precision (False)    0.8582180990722913
            ```
        - K-fold cross-validation
            ```
            --------------------------------------------------
            K-fold cross-validation
            --------------------------------------------------
            Confusion matrix
                                 Predict positive      Predict negative
            Actual positive      819.6666666666666     1022.6666666666666
            Actual negative      309.6666666666667     5445
            --------------------------------------------------
            Accuracy             0.8246237549910052
            Sensitivity (True)   0.4449068210602497
            Sensitivity (False)  0.9461886005560703
            Precision (True)     0.7257969303423849
            Precision (False)    0.8418801216306756
            ```

6. Comparison & Conclusion
    1. From the above results, we can conclude that Random Forest model results in higher accurary than Decision Tree model
    2. In Decision Tree model, K-fold cross-validation is slightly better than Holdout validation
    3. In Random Forest model, Holdout validation is better than K-fold cross-validation
    4. If I drop the continuous data when constructing Decision Tree model, the result seems to be a bit better, which is quite interesting and I don't know why.
        - Maybe my design for continuous data has some mistakes
        - Or maybe the continuous data can not really determines the target, which in other words is their relation is low

7. Kaggle Submission: **0.83890**
    ![Imgur](https://imgur.com/Zi89KlR.png)
    *This screenshot was taken on 2019/11/22 12:30am*

8. Bonus: Answering question **Optimization the tree** asked by **Cham Chen**
    ![Imgur](https://imgur.com/Sjzjis5.png)

Codes
===
Decision Tree with Holdout Validation
--
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

                #print(chd_data)
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

random.shuffle(rows)
test_range_s = 0
test_range_e = test_range_s + int(data_count * 0.3)

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

print('Building tree')    

node = {'type': 0}
tree.append(node)
build_tree(0, data, [])    


print('Done')

tp = 0
tn = 0
fp = 0
fn = 0

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
            tp += 1
        else:
            tn += 1
    else:
        if tree[node]['predict'] == 1:
            fp += 1
        else:
            fn += 1

print("--------------------------------------------------") 
print("Holdout Validation")
print("--------------------------------------------------")           
print("Confusion matrix")
print("                     Predict positive      Predict negative")
print("Actual positive     ", tp, "                 ", fn)
print("Actual negative     ", fp, "                 ", tn)
print("--------------------------------------------------")
print("Accuracy            ", (tp+tn)/(tp+tn+fp+fn))
print("Sensitivity (True)  ", (tp) / (tp+fn))
print("Sensitivity (False) ", (tn) / (fp+tn))
print("Precision (True)    ", (tp) / (tp+fp))
print("Precision (False)   ", (tn) / (tn+fn))
```

Decision Tree with K-fold cross-validation
--
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
```

Codes for generating output by Decision Tree to submit
--
```python=
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])

    with open('X_test.csv', newline='') as testfile:
        testrows = list(csv.reader(testfile))  

    for r in range(1, len(testrows)):
        testrow = testrows[r]
        node = 0

        while tree[node]['type'] != 2:
            sltd_key = tree[node]['sltd_key']
            if sltd_key in ctgl_features:
                node = tree[node][ testrow[sltd_key] ]
            else:
                value = tree[node]['value']
                node = tree[node][ int(int(testrow[sltd_key])<value) ]
        writer.writerow([testrow[0], tree[node]['predict']])
print('done')
```

Random Forest with Holdout Validation
--
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

feature_dict = {}
ctgl_features = [2, 4, 6, 7, 8, 9, 10, 14]
tree_count = 10
tree_feature_count = 8
tree_feature = []
forest = []
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

def build_tree(node_num, data, prt_sltd_key, tree_i):
    total_value = entropy(data)
    
    if (total_value < 0.02):
        tree[node_num]['type'] = 2
        tree[node_num]['predict'] = predict(data)
        return
    else:
        info_gain = -9999999
        sltd_key = -1
        sltd_value = -1
        for key in tree_feature[tree_i]:
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
                build_tree(chd_node_num, chd_data, prt_sltd_key, tree_i)
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
            build_tree(chd_node_num, s_chd_data, prt_sltd_key, tree_i)
            
            chd_node_num = len(tree)
            tree[node_num][0] = chd_node_num
            node = {'type': 1}
            tree.append(node)
            build_tree(chd_node_num, b_chd_data, prt_sltd_key, tree_i)
        
        prt_sltd_key.remove(sltd_key)
    return
        

#### Main ####
with open('y_train.csv', newline='') as trainfile:
    yrows = list(csv.reader(trainfile))   
    del yrows[0]

with open('X_train.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
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

random.shuffle(rows)
test_range_s = 0
test_range_e = test_range_s + int(data_count * 0.3)



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
    

for tree_i in range (0, tree_count):
    features = []
    while len(features) < tree_feature_count:
        new_feature = random.randint(1, 14)
        if new_feature not in features:
            features.append(new_feature)
    tree_feature.append(features)

    
    tree = []
    
    print('Building tree', tree_i)    

    node = {'type': 0}
    tree.append(node)
    build_tree(0, data, [], tree_i)    

    print('Done')
    
    forest.append(tree)

tp = 0
tn = 0
fp = 0
fn = 0

for r in range(test_range_s, test_range_e):
    row = rows[r]
    ans = {0: 0, 1: 0}
    pdct = 0
    
    for tree_i in range (0, tree_count):
        tree = forest[tree_i]
        node = 0

        while tree[node]['type'] != 2:
            sltd_key = tree[node]['sltd_key']
            if sltd_key in ctgl_features:
                node = tree[node][ row[sltd_key] ]
            else:
                value = tree[node]['value']
                node = tree[node][ int(int(row[sltd_key])<value) ]

        ans[ tree[node]['predict'] ] += 1
    
    if ans[1] > ans[0]:
        pdct = 1
    
    if pdct == int(row[len(row)-1]):
        if pdct == 1:
            tp += 1
        else:
            tn += 1
    else:
        if pdct == 1:
            fp += 1
        else:
            fn += 1

print("--------------------------------------------------") 
print("Holdout Validation")
print("--------------------------------------------------")           
print("Confusion matrix")
print("                     Predict positive      Predict negative")
print("Actual positive     ", tp, "                 ", fn)
print("Actual negative     ", fp, "                 ", tn)
print("--------------------------------------------------")
print("Accuracy            ", (tp+tn)/(tp+tn+fp+fn))
print("Sensitivity (True)  ", (tp) / (tp+fn))
print("Sensitivity (False) ", (tn) / (fp+tn))
print("Precision (True)    ", (tp) / (tp+fp))
print("Precision (False)   ", (tn) / (tn+fn))
```
Random Forest with K-fold cross-validation
--
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

feature_dict = {}
ctgl_features = [2, 4, 6, 7, 8, 9, 10, 14]
tree_count = 10
tree_feature_count = 8
tree_feature = []
forest = []
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

def build_tree(node_num, data, prt_sltd_key, tree_i):
    total_value = entropy(data)
    
    if (total_value < 0.02):
        tree[node_num]['type'] = 2
        tree[node_num]['predict'] = predict(data)
        return
    else:
        info_gain = -9999999
        sltd_key = -1
        sltd_value = -1
        for key in tree_feature[tree_i]:
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
                build_tree(chd_node_num, chd_data, prt_sltd_key, tree_i)
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
            build_tree(chd_node_num, s_chd_data, prt_sltd_key, tree_i)
            
            chd_node_num = len(tree)
            tree[node_num][0] = chd_node_num
            node = {'type': 1}
            tree.append(node)
            build_tree(chd_node_num, b_chd_data, prt_sltd_key, tree_i)
        
        prt_sltd_key.remove(sltd_key)
    return


#### Main ####
with open('y_train.csv', newline='') as trainfile:
    yrows = list(csv.reader(trainfile))   
    del yrows[0]

with open('X_train.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
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

    forest = []
    tree_feature = []
    
    for tree_i in range (0, tree_count):
        features = []
        while len(features) < tree_feature_count:
            new_feature = random.randint(1, 14)
            if new_feature not in features:
                features.append(new_feature)
        tree_feature.append(features)

        
        tree = []

        print('Building tree', tree_i)    

        node = {'type': 0}
        tree.append(node)
        build_tree(0, data, [], tree_i)    

        print('Done')

        forest.append(tree)

    for r in range(test_range_s, test_range_e):
        row = rows[r]
        ans = {0: 0, 1: 0}
        pcdt = 0

        for tree_i in range (0, tree_count):
            tree = forest[tree_i]
            node = 0

            while tree[node]['type'] != 2:
                sltd_key = tree[node]['sltd_key']
                if sltd_key in ctgl_features:
                    node = tree[node][ row[sltd_key] ]
                else:
                    value = tree[node]['value']
                    node = tree[node][ int(int(row[sltd_key])<value) ]

            ans[ tree[node]['predict'] ] += 1

        if ans[1] > ans[0]:
            pcdt = 1

        if pcdt == int(row[len(row)-1]):
            if pcdt == 1:
                tp[k_ford_i] += 1
            else:
                tn[k_ford_i] += 1
        else:
            if pcdt == 1:
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
```

Codes for generating output by Random Forest to submit
--
```python=
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])

    with open('X_test.csv', newline='') as testfile:
        testrows = list(csv.reader(testfile))  
    
    for r in range(1, len(testrows)):
        testrow = testrows[r]
        ans = {0: 0, 1: 0}
        pdct = 0

        for tree_i in range (0, tree_count):
            tree = forest[tree_i]
            node = 0

            while tree[node]['type'] != 2:
                sltd_key = tree[node]['sltd_key']
                if sltd_key in ctgl_features:
                    node = tree[node][ testrow[sltd_key] ]
                else:
                    value = tree[node]['value']
                    node = tree[node][ int(int(testrow[sltd_key])<value) ]

            ans[ tree[node]['predict'] ] += 1

        if ans[1] > ans[0]:
            pdct = 1
        
        writer.writerow([testrow[0], pdct])
print('done')
```

###### tags: `NCTU` `ML` `Decision Tree` `Random Forest`