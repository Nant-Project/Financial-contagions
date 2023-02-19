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
        filename = file.split("\\")[-1].replace(".csv", "")
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

def compute_log_return(dfDictist, index):
    close_price_i = dfDictist[index]['Close']
    log_returns=np.log(close_price_i/close_price_i.shift(1))
    return log_returns

def compute_log_returns(dfDictist,indexes_names):
    dfs = None
    for index in indexes_names:
        df = compute_log_return(dfDictist, index)
        if dfs is None:
            dfs = df[:]
        else:
            dfs = pd.concat([dfs, df], axis=1)
    dfs.columns = indexes_names
    dfs = dfs.loc[(dfs.index >= '2000-01-01')]
    #dfs = dfs.fillna(0)
    return dfs

def estimate_trials_of_same_sign(dfDictist,indexes_names,index1,index2):
    log_returns=compute_log_returns(dfDictist,indexes_names)
    Index_1 = dfDictist[index1]
    close_open_difference_1=Index_1['Close']-Index_1['Open']
    log_returns_2=log_returns[index2]
    dictionary_trials=dict()
    for i in log_returns_2.index:
        counter=0
        if i in close_open_difference_1.index:
            if log_returns_2.loc[i]<0 and close_open_difference_1.loc[i]<0:
                counter=counter+1
            if log_returns_2.loc[i]>0 and close_open_difference_1.loc[i]>0:
                counter=counter+1
            dictionary_trials[close_open_difference_1.loc[i]]=counter
    return dictionary_trials

r'''
def estimate_trials_of_same_sign(dfDictist,indexes_names,index1,index2):
    log_returns=compute_log_returns(dfDictist,indexes_names)
    Index_1 = dfDictist[index1]
    close_open_difference_1=np.abs(Index_1['Open']-Index_1['Close'])
    returns_close_open_difference=np.log(close_open_difference_1/close_open_difference_1.shift(1))
    returns_close_open_difference=returns_close_open_difference.dropna()
    log_returns_2=log_returns[index2]
    dictionary_trials=dict()
    for i in log_returns_2.index:
        counter=0
        if i in close_open_difference_1.index:
            if log_returns_2.loc[i]<0 and returns_close_open_difference.loc[i]<0:
                counter=counter+1
            if log_returns_2.loc[i]>0 and returns_close_open_difference.loc[i]>0:
                counter=counter+1
            dictionary_trials[returns_close_open_difference.loc[i]]=counter
    return dictionary_trials
'''
from datetime import datetime  
from datetime import timedelta 

def estimate_trials_of_same_sign(dfDictist,indexes_names,index1,index2):
    log_returns=compute_log_returns(dfDictist,indexes_names)
    Index_1 = dfDictist[index1]
    close_open_difference_1=Index_1['Close']/Index_1['Open']
    #close_open_difference_1=np.abs(Index_1['Close']-Index_1['Open'])
    returns_close_open_difference=np.log(close_open_difference_1/close_open_difference_1.shift(1))
    returns_close_open_difference=returns_close_open_difference.dropna()
    Index_2 = dfDictist[index2]
    #close_open_difference_2=np.abs(Index_2['Open']-Index_2['Close'])
    close_open_difference_2=Index_2['Open']/Index_2['Close']
    returns_close_open_difference_2=np.log(close_open_difference_2/close_open_difference_2.shift(1))
    returns_close_open_difference_2=returns_close_open_difference_2.dropna()
    dictionary_trials=dict()
    for i in log_returns.index:
        counter=0
        if i in returns_close_open_difference.index and i+timedelta(days=1) in returns_close_open_difference_2.index :
            if returns_close_open_difference_2.loc[i+timedelta(days=1)]<0 and returns_close_open_difference.loc[i]<0:
                counter=counter+1
            if returns_close_open_difference_2.loc[i+timedelta(days=1)]>0 and returns_close_open_difference.loc[i]>0:
                counter=counter+1
            dictionary_trials[returns_close_open_difference.loc[i]]=counter
    return dictionary_trials

def divide_data_into_chunks(dfDictist,indexes_names,index1,index2):
    trials=estimate_trials_of_same_sign(dfDictist,indexes_names,index1,index2)
    intervals=np.linspace(-0.07, 0.06, num=30)
    #intervals=[i for i in range(-9,9)]
    dict_probabilities=dict()
    for i in range(len(intervals)-1):
        ones=0
        observations=0
        for k in trials.keys():
            if k>=intervals[i] and k<=intervals[i+1]:
                observations=observations+1
                if trials[k]==1:
                    ones=ones+1
        if observations!=0:
            dict_probabilities[(intervals[i],intervals[i+1])]=ones/observations
    return(dict_probabilities)

def make_figure_1(dfDictist,indexes_names,index1,index2):
    dict_probabilities=divide_data_into_chunks(dfDictist,indexes_names,index1,index2)
    x_values=[i[0] for i in dict_probabilities.keys()]
    y_values=list(dict_probabilities.values())
    #plt.figure()
    if index2 in ['AEX','FCHI','GDAX','GPTSE','SSMI']:
        plt.scatter(x_values,y_values,c='blue',marker='o')
    if index2 in ['HSI','JKSE','KS11','N225','SSEC','TWII','TA125 (formerly TA100)']:
        plt.scatter(x_values,y_values,c='orange',marker='x')
    plt.xlabel('$R_{US \: open-close}$')
    plt.ylabel('Pr(sign($R_{i}$)=sign($R_{US \: open-close}$))')
    plt.title(index2)
    plt.show()

def plot_mean(dfDictist,indexes_names,index1):
    mean_Europe=np.zeros(29)
    mean_Asia=np.zeros(29)
    for index2 in indexes_names:
        dict_probabilities=divide_data_into_chunks(dfDictist,indexes_names,index1,index2)
        x_values=[i[1] for i in dict_probabilities.keys()]
        y_values=np.array(list(dict_probabilities.values()))
        if index2 in ['AEX','FCHI','GDAX','GPTSE','SSMI']:
            mean_Europe=mean_Europe+y_values
        if index2 in ['HSI','JKSE','KS11','N225','SSEC','TWII','TA125 (formerly TA100)']:
            mean_Asia=mean_Asia+y_values
    mean_Europe=mean_Europe/5
    mean_Asia=mean_Asia/7
    plt.scatter(x_values,mean_Europe,c='blue',marker='o',label='Europe')
    plt.scatter(x_values,mean_Asia,c='orange',marker='x',label='Asia')
    plt.legend()
    plt.xlabel('$R_{US \: open-close}$')
    plt.ylabel('Pr(sign($R_{i}$)=sign($R_{US \: open-close}$))')
    plt.ylim(0.3,1.0)
    #plt.savefig('Figure1.pdf')
    plt.show()
    
#Try to plot the 1st picture b):
#Use the capitalisation for year 2017 as it is available for most: 

def load_capitalisations(pathdir,indexes_names):
    capitalisations=pd.read_csv(pathdir+'\\Capitalisations.csv')
    country_names=['Netherlands','United States','Austria','Brazil','France','Germany','Canada','China','Switzerland','Mexico','Japan',"Korea, Dem. People's Rep.",
                   'Indonesia','Israel','Hong Kong SAR, China']
    index_names=['AEX','spx_d','ATX','BVSP','FCHI','GDAX','GSPTSE','SSEC','SSMI','MXX','N225','KS11','JKSE','TA125 (formerly TA100)','HSI']
    capitalisations_values=list()
    for country in country_names:
        capitalisations_values.append(capitalisations[capitalisations['Country Name']==country]['2017'].values[0])
    dict_capitalisations = {index_names[i]: capitalisations_values[i] for i in range(len(index_names))}
    for i in indexes_names:
        if i not in dict_capitalisations.keys():
            dict_capitalisations[i]=np.nan
    #Remove the nans:
    del dict_capitalisations['KS11']
    del dict_capitalisations['TWII']
    return(dict_capitalisations)

def calculate_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir):
    log_returns=compute_log_returns(dfDictist,indexes_names)
    dict_capitalisations=load_capitalisations(pathdir,indexes_names)
    R_ms=list()
    for date in log_returns.index:
        R_m=0
        for index in dict_capitalisations.keys():
            R_m=R_m+log_returns[index].loc[date]*dict_capitalisations[index]
        R_ms.append(R_m/np.sum([i for i in dict_capitalisations.values() if str(i)!='nan']))
    R_ms=pd.Series(R_ms,index=list(log_returns.index))
    return(R_ms)

def estimate_trials_of_same_sign_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2):
    log_returns=compute_log_returns(dfDictist,indexes_names)
    Index_1 = calculate_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir)
    log_returns_2=log_returns[index2]
    dictionary_trials=dict()
    for i in log_returns_2.index:
        counter=0
        if i in Index_1.index:
            if log_returns_2.loc[i]<0 and Index_1.loc[i]<0:
                counter=counter+1
            if log_returns_2.loc[i]>0 and Index_1.loc[i]>0:
                counter=counter+1
            dictionary_trials[Index_1.loc[i]]=counter
    return dictionary_trials

def divide_data_into_chunks_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2):
    trials=estimate_trials_of_same_sign_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2)
    intervals=[-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05,0.06]
    intervals=np.linspace(-0.06,0.07,50)
    dict_probabilities=dict()
    for i in range(len(intervals)-1):
        ones=0
        observations=0
        for k in trials.keys():
            if k>=intervals[i] and k<=intervals[i+1]:
                observations=observations+1
                if trials[k]==1:
                    ones=ones+1
        if observations!=0:
            dict_probabilities[(intervals[i],intervals[i+1])]=ones/observations
    return(dict_probabilities)

def make_figure_1a(dfDictist,indexes_names,dict_capitalisations,pathdir,index2):
    dict_probabilities=divide_data_into_chunks_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2)
    x_values=[i[0] for i in dict_probabilities.keys()]
    y_values=list(dict_probabilities.values())
    plt.figure()
    plt.scatter(x_values,y_values,c='blue',marker='o')
    plt.xlabel('$R_{m}$')
    plt.ylabel('Pr(sign($R_{i}$)=sign($R_{m}$))')
    plt.title(index2)
    plt.show()

def plot_mean_1a_per_continent(dfDictist,indexes_names,dict_capitalisations,pathdir):
    mean_Europe=np.zeros(37)
    mean_Asia=np.zeros(37)
    for index2 in indexes_names:
        dict_probabilities=divide_data_into_chunks_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2)
        x_values=[i[1] for i in dict_probabilities.keys()]
        y_values=np.array(list(dict_probabilities.values()))
        if index2 in ['AEX','FCHI','GDAX','GPTSE','SSMI']:
            mean_Europe=mean_Europe+y_values
        if index2 in ['HSI','JKSE','KS11','N225','SSEC','TWII','TA125 (formerly TA100)']:
            mean_Asia=mean_Asia+y_values
    mean_Europe=mean_Europe/5
    mean_Asia=mean_Asia/7
    plt.scatter(x_values,mean_Europe,c='blue',marker='o',label='Europe')
    plt.scatter(x_values,mean_Asia,c='orange',marker='x',label='Asia')
    plt.legend(loc='lower right')
    plt.xlabel('$R_{m}$')
    plt.ylabel('Pr(sign($R_{i}$)=sign($R_{m}$))')
    plt.ylim(0.3,1.1)
    plt.savefig('Figure1a_divided_by_continent.pdf')
    plt.show()  
    
def plot_mean_1a(dfDictist,indexes_names,dict_capitalisations,pathdir):
    mean_values=np.zeros(37)
    for index2 in indexes_names:
        dict_probabilities=divide_data_into_chunks_Rm(dfDictist,indexes_names,dict_capitalisations,pathdir,index2)
        x_values=[i[1] for i in dict_probabilities.keys()]
        y_values=np.array(list(dict_probabilities.values()))
        mean_values=mean_values+y_values
    mean_values=mean_values/len(indexes_names)
    plt.scatter(x_values,mean_values,c='blue',marker='o')
    plt.xlabel('$R_{m}$')
    plt.ylabel('Pr(sign($R_{i}$)=sign($R_{m}$))')
    plt.ylim(0.3,1.1)
    plt.savefig('Figure1a_2.pdf')
    plt.show() 

    
    
        
        