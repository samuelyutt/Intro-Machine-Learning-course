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