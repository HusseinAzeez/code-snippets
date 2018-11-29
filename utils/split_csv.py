csvfile = open('./training3.csv', 'r').readlines()
filename = 1
for i in range(len(csvfile)):
    if i % 5000 == 0:
        open(str(filename) + '.csv', 'w+').writelines(csvfile[i:i+5000])
        filename += 1
