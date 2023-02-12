import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json
from functools import reduce
from datetime import datetime


def load_indexes(pathdir):
    assert os.path.exists(pathdir), "file not exists: "+pathdir
    files = glob.glob(pathdir+"/*.csv")
    dfDictist = dict()
    indexes_names = []
    for file in files:
        filename = file.split("/")[-1].replace(".csv", "")
        indexes_names.append(filename)
        dfDictist[filename] = pd.read_csv(
            file, parse_dates=[0]).set_index('Date')
    return dfDictist, indexes_names


def betaFunction(i_index, j_index, Z,
                 tho): return np.exp(-abs(Z[i_index]-Z[j_index])/tho)


def alphaFunction(i_index, j_index, K, gamma): return 1 - \
    np.exp(-K[j_index]/(K[i_index]*gamma))


def compute_beta_matrix(timezonePath, indexes_names, tho=3):
    n = len(indexes_names)
    beta = np.zeros((n, n))
    with open(timezonePath) as file:
        Z = json.load(file)["timezone"]
    for i, i_index in enumerate(indexes_names):
        for j, j_index in enumerate(indexes_names):
            beta[i, j] = betaFunction(i_index, j_index, Z, tho)
    return beta


def compute_alpha_matrix(CapitalisationPath, indexes_names, gamma, filed="UniformCapitalisation"):
    n = len(indexes_names)
    alpha = np.zeros((n, n))
    with open(CapitalisationPath) as file:
        K = json.load(file)[filed]
    for i, i_index in enumerate(indexes_names):
        for j, j_index in enumerate(indexes_names):
            alpha[i, j] = alphaFunction(i_index, j_index, K, gamma)
    return alpha


def intersection_function(s1, s2): return s1.intersection(s2)


def common_dates(dfDictist):
    return sorted(reduce(common_dates, list(map(lambda df: set(df.Date.values), dfDictist.values()))))


def covert_to_datetime(dates):
    return list(map(lambda string: datetime.strptime(string, "%Y-%m-%d")), dates)


def compute_eta(dfDictist, index):
    Index_i = dfDictist[index]
    return (Index_i["Close"]-Index_i["Open"])


def comupte_etas(dfDictist, index_names):
    dfs = None
    for index in index_names:
        df = compute_eta(dfDictist, index)
        if dfs is None:
            dfs = df[:]
        else:
            dfs = pd.concat([dfs, df], axis=1)
    dfs.columns = index_names
    #dfs=dfs.loc[(dfs.index >= '2009-01-01') & (dfs.index <= '2010-01-01')]
    dfs = dfs.loc[(dfs.index >= '2000-01-01')]
    # dfs=dfs.dropna()
    dfs = dfs.fillna(0)
    return dfs


def Heaviside(r, r_critic): return 1*(r >= r_critic)


def compute_Returns(dfDictist, index_names, r_critic, CapitalisationPath, indexes_names, gamma):
    etas = comupte_etas(dfDictist, index_names)
    alpha = compute_alpha_matrix(CapitalisationPath, indexes_names, gamma)
    beta = compute_beta_matrix(CapitalisationPath, indexes_names)
    dates = list(etas.index)
    P = pd.DataFrame(0, index=etas.index, columns=etas.columns)
    R = pd.DataFrame(0, index=etas.index, columns=etas.columns)
    Rcum = pd.DataFrame(0, index=etas.index, columns=etas.columns)
    N = len(index_names)
    for t, date in enumerate(dates[1:]):
        R.loc[date] = etas.iloc[t] + \
            (alpha*beta)@(Heaviside(Rcum.iloc[t-1], r_critic)*Rcum.iloc[t-1])/N
        Rcum.loc[date] = (1-Heaviside(Rcum.iloc[t-1], r_critic)
                          )*(Rcum.iloc[t-1]+R.iloc[t])
    return R, Rcum
