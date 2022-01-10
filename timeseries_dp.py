#%%
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from argparse import Namespace
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool, freeze_support

args = Namespace(
    original_file = 'original file name',
    output_file = 'output file name',
    epsilon = 'choose epsilon between 0.1 ~ 10000',
    index = 'person_index',
    columns = 'timeline column name',
    variable = 'variable you want to synthesize'
)



#%%
## timeseries DP algorithm 

# I. Preprocessing
# input : original Dataset with time steps(columns), patients as index

## 1. get the max time step
raw_timeseries = pd.read_csv(args.original_file)
max_timestep = raw_timeseries.month.max()

#%%
## 2. change the DataFrame. use pivot table. save the Na information
df = deepcopy(raw_timeseries.pivot_table(index=args.index, columns = args.columns, values=args.variable))
df_frame = deepcopy(df) # for every iterations we are going to fill this frame
na_information = ~df.isna()
original_idx = df.columns

#%%
## 3. Fill Na values and 
df = df.interpolate(method='linear', limit_direction='forward',axis=1)
df.columns = [i for i in range(0,len(df.columns))]

#%%

# II. TimeSeries Differential Privacy
# input : processed original Data

## 0. define laplace mechanism
def get_differential_privacy_value(value, epsilon):
    
    np.array(value)
    def pdf(x):
        b = 2 / (epsilon)
        c = 1 - 0.5 * (np.exp(-(value+1)/b) + np.exp(-(1 - value)/b))
        return 1 / (b * c * 2) * np.exp(-np.absolute(x - value)/b)

    elements = np.linspace(-1, 1, 10**4)
    probabilities = pdf(elements)
    probabilities /= np.sum(probabilities)
    return np.random.choice(elements, size=1, p=probabilities.reshape(-1)).item()


#%%
## 1. get synthetic value v1'
epsilon = args.epsilon

    
def timeseries_dp(timeseries, epsilon):
    print('timeseries_dp activated')
    scaler = MinMaxScaler(feature_range=(-1,1))
    
    S = [] # for the values in the sample
    
    for val_idx, value in enumerate(timeseries):
        if val_idx == 0 :
            # 시작점에서는 값이 두개뿐
            ranges = np.array([value,timeseries[val_idx+1]]).reshape(-1,1)
            scaler.fit(ranges)
            scaled_value = scaler.transform(ranges)[0]
            dp_value = get_differential_privacy_value(scaled_value, epsilon) # 여기를 나중에 바꿔줄 것임
            # synthesized value : v'
            dp_value = np.array(dp_value).reshape(-1,1)
            print(dp_value)
            syn_value = scaler.inverse_transform(dp_value)
            S.append(syn_value.item())
            
        
## 2. get all values after index 0
        elif val_idx == len(timeseries)-1 :
            ranges = np.array([value,timeseries[val_idx-1]]).reshape(-1,1)
            scaler.fit(ranges)
            scaled_value = scaler.transform(ranges)[0]
            dp_value = get_differential_privacy_value(scaled_value, epsilon)
            dp_value = np.array(dp_value).reshape(-1,1)
            syn_value = scaler.inverse_transform(dp_value)
            S.append(syn_value.item())
            
        else :
            v_1, v_3 = timeseries[val_idx-1], timeseries[val_idx+1] # value index +1 을 v_3로 표현
            if (v_1 < value < v_3) | (v_1 > value > v_3) :
                ranges = np.array([v_1, v_3]).reshape(-1,1)
                scaler.fit(ranges)
                scaled_value = scaler.transform(np.array(value).reshape(-1,1))
                dp_value = get_differential_privacy_value(scaled_value, epsilon)
                dp_value = np.array(dp_value).reshape(-1,1)
                syn_value = scaler.inverse_transform(dp_value)
                S.append(syn_value.item())
                
            
            elif (v_1 < value) & (v_3 < value):
                _, second, first = sorted([v_1, value, v_3])
                ranges = np.array([first, second]).reshape(-1,1)
                scaler.fit(ranges)
                scaled_value = scaler.transform(np.array(value).reshape(-1,1))
                dp_value = get_differential_privacy_value(scaled_value, epsilon)
                dp_value = np.array(dp_value).reshape(-1,1)
                syn_value = scaler.inverse_transform(dp_value)
                S.append(syn_value.item())
                
            elif (v_1 > value) & (v_3 > value) :
                third, second, _ = sorted([v_1, value, v_3])
                ranges = np.array([third, second]).reshape(-1,1)
                scaler.fit(ranges)
                scaled_value = scaler.transform(np.array(value).reshape(-1,1))
                dp_value = get_differential_privacy_value(scaled_value, epsilon)
                dp_value = np.array(dp_value).reshape(-1,1)
                syn_value = scaler.inverse_transform(dp_value)
                S.append(syn_value.item())
                
            else :
                if (value == v_1 == v_3) :
                    # syn_value = get_differential_privacy_value(value, epsilon)
                    syn_value = 0
                    S.append(syn_value)
                else : 
                    val1, val2 = list(set([value, v_1, v_3]))
                    ranges = np.array([val1, val2]).reshape(-1,1)
                    scaler.fit(ranges)
                    scaled_value = scaler.transform(np.array(value).reshape(-1,1))
                    dp_value = get_differential_privacy_value(scaled_value, epsilon)
                    dp_value = np.array(dp_value).reshape(-1,1)
                    syn_value = scaler.inverse_transform(dp_value)
                    S.append(syn_value.item())
    return S


#%% 
# split the data into rows
lst = [(sample,epsilon) for smple_idx, (name, sample) in enumerate(df.iterrows())]
names = [pt for pt,_ in df.iterrows()]

#%%
p = Pool(10)
output = p.starmap(timeseries_dp,lst)
print('processing done')
    
output = np.array(output)

for pt, row in zip(names,output):
    df_frame.loc[pt] = row


#%%
# 원본 데이터 형식으로 회복
df_frame = df_frame[na_information]
df_frame = df_frame.reset_index()

# df_Frame melt. raw_timeseries 형태로
df_frame = pd.melt(df_frame, id_vars=args.index,
        value_vars=list(df_frame.columns[1:]),
        var_name=args.columns,
        value_name=args.variable)
df_frame = df_frame.dropna().reset_index(drop=True)
df_frame.columns = [args.index, args.columns, args.variable]
df_frame.to_csv(args.output_file, header=True, index=False, sep=',')
# %%
print('finished..!')
        