import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json


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

betaFunction=lambda i_index,j_index,Z,tho:np.exp(-abs(Z[i_index]-Z[j_index])/tho)
alphaFunction=lambda i_index,j_index,K,gamma:1-np.exp(-K[j_index]/(K[i_index]*gamma))

def compute_beta_matrix(timezonePath,indexes_names,tho=3):
    n=len(indexes_names)
    beta=np.zeros((n,n))
    with open(timezonePath) as file:
        Z=json.load(file)["timezone"]
    for i,i_index in enumerate(indexes_names):
        for j,j_index in enumerate(indexes_names):
            beta[i,j]=betaFunction(i_index,j_index,Z,tho)
    return beta

def compute_alpha_matrix(CapitalisationPath,indexes_names,gamma,filed="UniformCapitalisation"):
    n=len(indexes_names)
    alpha=np.zeros((n,n))
    with open(CapitalisationPath) as file:
        K=json.load(file)[filed]
    for i,i_index in enumerate(indexes_names):
        for j,j_index in enumerate(indexes_names):
            alpha[i,j]=alphaFunction(i_index,j_index,K,gamma)
    return alpha
