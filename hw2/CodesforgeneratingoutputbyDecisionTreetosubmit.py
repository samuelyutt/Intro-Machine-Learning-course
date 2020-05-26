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