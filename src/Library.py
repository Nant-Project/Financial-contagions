import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def load_indexes(pathdir):
    assert os.path.exists(pathdir),"file not exists"
    files = glob.glob(pathdir+"/*.csv")
    dfDictist=dict()
    indexes_names=[]
    for file in files:
        filename=file.split("/")[-1].replace(".csv","")
        indexes_names.append(filename)
        dfDictist[filename]=pd.read_csv(file)
    return dfDictist,indexes_names


