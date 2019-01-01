# Standard librray imports
import glob
import os

# Third-party imports
import pandas as pd
from tqdm import tqdm


def concatenate(outfile):
    """Concatenate one or multiple cvs files into one single file.

    Args:
        outfile (str): The output file path and name.
    Returns:
        None.
    Raises:
        TypeError: if n is not a number.
        ValueError: if n is negative.
    """
    fileList = glob.glob('../mnist-in-csv/Training set 3/*.csv')
    dfList = []
    colnames = []
    colnames.append('label')
    for no in range(0, 784):
        colnames.append('pixel ' + str(no))

    for filename in tqdm(fileList):
        df = pd.read_csv(filename, header=None)
        dfList.append(df)
    concatDf = pd.concat(dfList, axis=0)
    concatDf.columns = colnames
    concatDf.to_csv(outfile, index=False, encoding='utf-8')


if __name__ == '__main__':
    concatenate('../datasets/full_mix.csv')
