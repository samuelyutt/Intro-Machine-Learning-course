2019 Introduction to Machine Learning Program
Assignment #3 - Linear Regression and Logistic Regression
===
:::info
CS10 游騰德 0616026
:::

Linear Regression
===
Procedures of calculating A inverse
--
1. Calculate `A transpose`
    ```python=
    AT = []
    ATrow = []
    for i in range( len(A[0]) ):
        ATrow.append([])

    for Arow in A:
        for i in range( len(A[0]) ):
            ATrow[i].append(Arow[i])

    for i in range( len(A[0]) ):        
        AT.append(ATrow[i])
    ```
2. Calculate `A transpose` * `A`
    ```python=
    ATA = []
    for row in range( len(A[0]) ):
        ATArow = []
        for col in range( len(A[0]) ):
            num = 0
            for j in range( len(AT[0]) ):
                num += ( AT[row][j] * A[j][col] )
            ATArow.append(num)
        ATA.append(ATArow)
    ```
3. Calculate the `inverse of A transpose * A` by Gauss-Jordan elimination
    ```python=
    ATAinverse = []
    tempATA = []

    for i in range( len(ATA) ):
        ATAinverserow = [];
        tempATArow = [];
        for j in range( len(ATA) ):
            ATAinverserow.append( int(i==j) )
            tempATArow.append( ATA[i][j] )
        ATAinverse.append(ATAinverserow)
        tempATA.append(tempATArow)

    for i in range( len(tempATA) ):
        dnmt = tempATA[i][i]
        for j in range( len(tempATA) ):
            tempATA[i][j] /= dnmt
            ATAinverse[i][j] /= dnmt

        for k in range(i+1, len(tempATA)):
            ratio = -tempATA[k][i]
            for j in range( len(tempATA) ):
                tempATA[k][j] += ( tempATA[i][j] * ratio )
                ATAinverse[k][j] += ( ATAinverse[i][j] * ratio )

    for i in range( len(tempATA)-1, -1, -1 ):
        dnmt = tempATA[i][i]
        for j in range( len(tempATA)-1, -1, -1  ):
            tempATA[i][j] /= dnmt
            ATAinverse[i][j] /= dnmt

        for k in range(i-1, -1, -1):
            ratio = -tempATA[k][i]
            for j in range( len(tempATA)-1, -1, -1 ):
                tempATA[k][j] += ( tempATA[i][j] * ratio )
                ATAinverse[k][j] += ( ATAinverse[i][j] * ratio )
    ```
4. Calculate `inverse of A transpose * A` * `A transpose`
    ```python=
    ATAinverseAT = []
    for row in range( len(ATAinverse) ):
        ATAinverseATrow = []
        for col in range( len(AT[0]) ):
            num = 0
            for j in range( len(ATAinverse[0]) ):
                num += ( ATAinverse[row][j] * AT[j][col] )
            ATAinverseATrow.append(num)
        ATAinverseAT.append(ATAinverseATrow)
    ```
Results
--
1. `n` = 2
    ```
    Fitting line: 2.129097066511054X^1 + -40.27912612124099
    Total error: 63987.84575778298
    ```
    ![imgur](https://imgur.com/WRjpms0.png)
2. `n` = 3
    ```
    Fitting line: -5.706656439716334X^2 + 1.4315153833201295X^1 + 8.999492270942902
    Total error: 2.778052563826315e-26
    ```
    ![imgur](https://imgur.com/cbt3HxX.png)
3. `n` = 8
    ```
    Fitting line: 1.3636866759293791e-55X^7 + -9.280574919515993e-33X^6 + 1.766857788413267e-21X^5 + -6.442329308908867e-16X^4 + 2.371991492111647e-13X^3 + -5.706656439720301X^2 + 1.4315153833198457X^1 + 8.999492270956443
    Total error: 5.1695982941603906e-20
    ```
    ![imgur](https://imgur.com/CvwEwi2.png)

Logistic Regression
===
Inputs
--
1. Logistic_data1
    ![imgur](https://imgur.com/oaheyCR.png)
2. Logistic_data2
    ![imgur](https://imgur.com/r0VRfPy.png)

Procedures
--
1. Inner products and Sigmoid function
    1. Inner products
        ```python=
        def innerprdt(a, b):
            prdt = 0;
            for i in range( len(a) ):
                prdt += (a[i] * b[i])
            return prdt
        ```
    2. Sigmoid function
        ```python=
        def sigmoid(wt, x):
            try:
                num = math.exp( -innerprdt(wt, x) )
            except OverflowError:
                num = float('inf')
            return 1/(1+num)
        ```
2. Methods
    1. L2-norm
        ```python=
        wp[j] += (0 - sigmoid(w, vxs[i])) * sigmoid(w, vxs[i]) * (1 - sigmoid(w, vxs[i])) * vxs[i][j]
        ```
    2. Cross Entropy
        ```python=
        wp[j] += (sigmoid(w, vxs[i]) - 0) * vxs[i][j]
        ```
3. Weights
    ```python=
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
    ```
    
Results
--
1. Logistic_data1
    1. L2-norm
        ```
        --------------------------------------------------
        Method               1
        Weight               [-5.135898670937125, 0.568115314222881, 0.444448019171329]
        --------------------------------------------------
        Confusion matrix
                             Is cluster 1        Is cluster 2
        Predict cluster 1    50                  0
        Predict cluster 2    0                   50
        --------------------------------------------------
        Precision            1.0
        Recall               1.0
        ```
        ![imgur](https://imgur.com/3QyksWO.png)
    2. Cross Entropy
        ```
        --------------------------------------------------
        Method               2
        Weight               [-2.7557585210096356, 0.32228962702943903, 0.3703344520718077]
        --------------------------------------------------
        Confusion matrix
                             Is cluster 1        Is cluster 2
        Predict cluster 1    50                  0
        Predict cluster 2    0                   50
        --------------------------------------------------
        Precision            1.0
        Recall               1.0
        ```
        ![imgur](https://imgur.com/v4bVKn3.png)
2. Logistic_data2
    1. L2-norm
        ```
        --------------------------------------------------
        Method               1
        Weight               [-4.248605295738713, 1.2062111170728038, 1.5989365819856387]
        --------------------------------------------------
        Confusion matrix
                             Is cluster 1         Is cluster 2
        Predict cluster 1    37                   8
        Predict cluster 2    13                   42
        --------------------------------------------------
        Precision            0.7636363636363637
        Recall               0.84
        ```
        ![imgur](https://imgur.com/33hVGlt.png)
    2. Cross Entropy
        ```
        --------------------------------------------------
        Method               2
        Weight               [-1.5141807568184031, 0.4624407600053358, 0.5213325472579626]
        --------------------------------------------------
        Confusion matrix
                             Is cluster 1         Is cluster 2
        Predict cluster 1    37                   8
        Predict cluster 2    13                   42
        --------------------------------------------------
        Precision            0.7636363636363637
        Recall               0.84
        ```
        ![imgur](https://imgur.com/Hqb2rOa.png)
        
Codes
===
Linear Regression
--
```python=
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import csv
import math
import random
import statistics
import matplotlib.pyplot as plt

with open('HW3_data/linear_data.txt', newline='') as csvfile:
    datas = list(csv.reader(csvfile))
    
#### Settings ####
n = 1
savefig = 0

A = []
b = []
for data in datas:
    Arow = []
    Arow.append(1)
    
    num = float(data[0])
    for i in range(1, n):
        Arow.append(num)
        num *= num
    
    A.append(Arow)
    b.append( float(data[1]))
    
AT = []
ATrow = []
for i in range( len(A[0]) ):
    ATrow.append([])

for Arow in A:
    for i in range( len(A[0]) ):
        ATrow[i].append(Arow[i])

for i in range( len(A[0]) ):        
    AT.append(ATrow[i])
    
ATA = []

for row in range( len(A[0]) ):
    ATArow = []
    for col in range( len(A[0]) ):
        num = 0
        for j in range( len(AT[0]) ):
            num += ( AT[row][j] * A[j][col] )
        ATArow.append(num)
    ATA.append(ATArow)

ATAinverse = []
tempATA = []

for i in range( len(ATA) ):
    ATAinverserow = [];
    tempATArow = [];
    for j in range( len(ATA) ):
        ATAinverserow.append( int(i==j) )
        tempATArow.append( ATA[i][j] )
    ATAinverse.append(ATAinverserow)
    tempATA.append(tempATArow)


# for i in range( len(tempATA) ):
#     print(tempATA[i])
# for i in range( len(ATAinverse) ):
#     print(ATAinverse[i])
# print()

    
for i in range( len(tempATA) ):
    dnmt = tempATA[i][i]
    for j in range( len(tempATA) ):
        tempATA[i][j] /= dnmt
        ATAinverse[i][j] /= dnmt
    
    for k in range(i+1, len(tempATA)):
        ratio = -tempATA[k][i]
        for j in range( len(tempATA) ):
            tempATA[k][j] += ( tempATA[i][j] * ratio )
            ATAinverse[k][j] += ( ATAinverse[i][j] * ratio )
        
# for i in range( len(tempATA) ):
#     print(tempATA[i])
# for i in range( len(ATAinverse) ):
#     print(ATAinverse[i])
# print()

for i in range( len(tempATA)-1, -1, -1 ):
    dnmt = tempATA[i][i]
    for j in range( len(tempATA)-1, -1, -1  ):
        tempATA[i][j] /= dnmt
        ATAinverse[i][j] /= dnmt
        
    for k in range(i-1, -1, -1):
        ratio = -tempATA[k][i]
        for j in range( len(tempATA)-1, -1, -1 ):
            tempATA[k][j] += ( tempATA[i][j] * ratio )
            ATAinverse[k][j] += ( ATAinverse[i][j] * ratio )
        
# for i in range( len(tempATA) ):
#     print(tempATA[i])
# for i in range( len(ATAinverse) ):
#     print(ATAinverse[i])
# print()

ATAinverseAT = []

for row in range( len(ATAinverse) ):
    ATAinverseATrow = []
    for col in range( len(AT[0]) ):
        num = 0
        for j in range( len(ATAinverse[0]) ):
            num += ( ATAinverse[row][j] * AT[j][col] )
        ATAinverseATrow.append(num)
    ATAinverseAT.append(ATAinverseATrow)

w = []

for row in range( len(ATAinverseAT) ):
    num = 0
    for j in range( len(ATAinverseAT[0]) ):
        num += ( ATAinverseAT[row][j] * b[j] )
    w.append(num)
    
print("Fitting line: ", end = '')
for i in range(n-1, -1, -1):
    print(w[i], end = '')
    if i > 0:
        print("X^", end = '')
        print(i, end = '')
        print(" + ", end = '')
print()

totalerr = 0;
for data in datas:
    pdctvalue = 0;
    tempX = 1
    for i in range(n):
        pdctvalue += w[i] * tempX
        tempX *= float(data[0])
    totalerr += ( pdctvalue - float(data[1]) ) * ( pdctvalue - float(data[1]) )
print("Total error:", totalerr)

xs = []
ys = []
for row in datas:
    xs.append(float(row[0]))
    ys.append(float(row[1]))
plt.plot(xs, ys, 'ro')

x = np.linspace(-5, 5, 256, endpoint = True)
y = 0;
tempX = 1
for i in range(n):
    y += w[i] * tempX
    tempX *= x
plt.plot(x, y)

if savefig:
    plt.savefig("Outputs/1_n="+str(n)+".png")
plt.show()
```

Logistic Regression
--
```python=
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
```

        
###### tags: `NCTU` `ML` `Linear Regression` `Logistic Regression`