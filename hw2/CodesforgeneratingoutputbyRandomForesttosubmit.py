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