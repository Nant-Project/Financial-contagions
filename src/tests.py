import numpy as np
from Library import *
import matplotlib.pyplot as plt


dfDictist,indexes_names=load_indexes("Data/indexes")
dfs=comupte_etas(dfDictist,indexes_names)
indexes_names=indexes_names[:2]
CapitalisationPath="Data/cleanData/TestData.json"
etas=comupte_etas(dfDictist,indexes_names)
gamma=0.5
alpha=compute_alpha_matrix(CapitalisationPath,indexes_names,gamma)
beta=compute_beta_matrix(CapitalisationPath,indexes_names)
dates=list(etas.index)
P= pd.DataFrame(0, index=etas.index, columns=etas.columns)
R = pd.DataFrame(0, index=etas.index, columns=etas.columns)
Rcum = pd.DataFrame(0, index=etas.index, columns=etas.columns)
N=len(indexes_names)    
r_critic=100
R,cum=compute_Returns(dfDictist,indexes_names,r_critic,CapitalisationPath,indexes_names,gamma)

R["SSEC"].plot()