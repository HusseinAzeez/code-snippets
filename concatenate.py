
import pandas as pd
import glob
import os


def concatenate(outfile="./training.csv"):
    fileList = glob.glob('../mnist-in-csv/*.csv')
    dfList = []
    colnames = []
    colnames.append('label')
    for no in range(0, 784):
        colnames.append('pixel ' + str(no))

    for filename in fileList:
        df = pd.read_csv(filename, header=None)
        dfList.append(df)
    concatDf = pd.concat(dfList, axis=0)
    concatDf.columns = colnames
    concatDf.to_csv(outfile, index=False, encoding='utf-8')


concatenate()

df = pd.read_csv('./training.csv')
print(df.iloc[:-1])
