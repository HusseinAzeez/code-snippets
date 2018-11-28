import pandas as pd
import glob
import os
import time
from tqdm import tqdm


def concatenate(outfile="./training3-full.csv"):
    fileList = glob.glob('../mnist-in-csv/Training set 3/*.csv')
    dfList = []
    colnames = []
    colnames.append('label')
    for no in range(0, 784):
        colnames.append('pixel ' + str(no))

    for filename in tqdm(fileList):
        df = pd.read_csv(filename, header=None)
        dfList.append(df)
        time.sleep(1)
    concatDf = pd.concat(dfList, axis=0)
    concatDf.columns = colnames
    concatDf.to_csv(outfile, index=False, encoding='utf-8')


concatenate()

df = pd.read_csv('./training3-full.csv')
print(df.shape)
