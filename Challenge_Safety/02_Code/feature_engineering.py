

import itertools

import numpy as np
import pandas as pd
import datetime
from keras.preprocessing.sequence import pad_sequences


def quantile_00(x, q=0.9):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.quantile(x, q)


def get_length_sequences_where_00(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return max(res) if len(res) > 0 else 0
    

def get_length_sequences_last_00(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res[-1] if len(res) > 0 else 0
    



def generic_features_creation_01(df):
    """
    Calculate Generic Trip Features 
    :param x:features 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    
    df['acceleration']  =  np.sqrt(df.acceleration_x*df.acceleration_x+\
                                 df.acceleration_y*df.acceleration_y+\
                                 df.acceleration_z*df.acceleration_z)
    df['gyro']  =  np.sqrt(df.gyro_x*df.gyro_x+\
                                 df.gyro_y*df.gyro_y+\
                                 df.gyro_z*df.gyro_z)
    df['stop'] =  np.where(df.Speed <2*10/36,1,0)
    df['signal_weak'] = np.where(df.Speed<0,1,0)
    #features['second_lag1']
    df['trip_start'] = df.bookingID.shift(+1)
    df['trip_start'] = np.where(df['bookingID'] ==df['trip_start'],0,1)
    
    df['trip_end'] = df.bookingID.shift(-1)
    df['trip_end'] = np.where(df['bookingID'] ==df['trip_end'],0,1)
    
    df['Speed_lag1'] = df.Speed.shift(+1)
    df['missing_ind'] = df.second.shift(+1)
    df['missing_ind'] = np.where((df['second'] - df['missing_ind'])<=2,0,1)
    df['missing_ind'] = np.where(df['trip_start']==1,0,df['missing_ind'])
    
    df['acc_cal'] = df.Speed - df.Speed_lag1
    df['acc_cal'] = np.where((df['trip_start'] + df['missing_ind'] +df['signal_weak'])>=1,0,df['acc_cal'])
   
    return df


def generic_trip_features_02(df):
    
    """
    Calculate Missing and Waek Signal Data
    :param x:features 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    df['signal_week'] = np.where(df.Speed<0,1,0)

    res = df.groupby(['bookingID'])['second'].agg({"max","count"}).reset_index()
    res.rename(columns = {"max":"trip_dur","count":"trip_data_rec"}, inplace=True)
    res['trip_dur']+=1
    res['trip_data_missing'] = res['trip_dur'] - res['trip_data_rec']
    
    res_rev = df.groupby(['bookingID'])['signal_week'].agg({"sum"}).reset_index()
    res_rev.rename(columns = {"sum":"trip_signal_weak"}, inplace=True)
    
    res = res.merge(res_rev, on = ['bookingID'], how="left")
    
    res['trip_data_missing_f'] = res['trip_data_missing']/res['trip_dur']
    res['trip_signal_weak_f'] = res['trip_signal_weak']/res['trip_data_rec']
    return res

def window_features_creation1_03(df, window_size=10, over_wd=5,
                        cols =['Accuracy','Bearing','Speed','acceleration','gyro','acc_cal','signal_weak','stop',
                               'trip_start','missing_ind','trip_end'],
                       metrics ={"mean","std","min","max"} ):
    
    """
    Agg. features at sliding window of 10s and overlap window of 5s
    :window_size = window size
    :over_wd = overlap window
    :cols = Agg. Columns
    :metrics: statistical Metrics calculated
    :param x:features 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
        
    start = datetime.datetime.now().replace(microsecond=0)
    
    bids = df.bookingID.unique(); no_bids = len(bids)
    no_cols = len(cols)
    metrics = sorted(metrics)
    
    res = []
    res_window1 = []; res_window2 = []
    i1 = 1; i2=0
    for i in range(no_bids):
        i1 = 1; i2=0;window1=[];window2=[]
        bid = bids[i]
        w1 = df[df.bookingID==bid]
        if len(w1)>=over_wd:
            w2 = df[df.bookingID==bid][over_wd:]
            nps2 = int(len(w2)/window_size)
            window2 = [i2]*over_wd
            i2+=2
            for j in range(0,nps2):
                b = [i2]*window_size; i2+=2
                window2+=b
                #print (window2)
            b = [i2]*len(w2[nps2*window_size:]); window2+=b
            res_window2+=window2
        else:
            res_window2+= [0]*len(w1)
            
        nps1 = int(len(w1)/window_size)
        for j in range(0,nps1):
            a = [i1]*window_size; i1+=2
            window1+=a
        a = [i1]*len(w1[nps1*window_size:]); window1+=a
        res_window1+=window1
   
    df['window1'] = res_window1
    df['window2'] = res_window2
    df_1 = df.groupby(['bookingID',"window1"])[cols].agg(metrics).reset_index()
    df_2 = df.groupby(['bookingID',"window2"])[cols].agg(metrics).reset_index()
    
    res_cols = ['bookingID','window']
    for col in cols:
        for m in metrics:
            res_cols.append(col+"_" + m)
    df_1.columns = res_cols
    df_2.columns = res_cols
    
    df = pd.concat([df_1,df_2], axis=0)
    df = df[df['window']>0]
    df = df.sort_values(by =['bookingID','window'])
    
    end = datetime.datetime.now().replace(microsecond=0)
    #print ("Sliding Windows Features Done --- >", end-start)
    return df

def window_grp_stop_04(df, cols = ['stop_sum'], thres=8):
     
    """
    Agg. features at sliding window of 10s and overlap window of 5s
    :cols = Agg. Columns
    :thres: threshold where we consider window as stop event 
    :param x:window_feas 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    
    df['ind'] = np.where(df[cols]>thres,1,0)
    df = df.groupby(['bookingID'])['ind'].agg({'sum','mean',get_length_sequences_where_00}).reset_index()
    df.rename(columns = {"sum":"stop_cnt","mean":"stop_mean","get_length_sequences_where_00":"stop_longest"}, inplace = True)
    return df

def window_grp_speed_05(df, cols = ['Speed_mean'], thres=8, metrics = {"max","mean"}):
    
    """
    Agg. features at sliding window of 10s and overlap window of 5s
    :cols = Agg. Columns
    :thres: threshold where we consider window as Speed event 
    :param x:window_feas 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    
    metrics = sorted(metrics)
    df['stop_ind'] = np.where(df["stop_sum"]>thres,1,0)
    df['ind'] = df['missing_ind_max'] + df['signal_weak_max'] + df['stop_ind'] 
    
    df1 = df.groupby(['bookingID'])[cols[0]].agg(metrics).reset_index()
    df1.columns = ['bookingID'] + ['speed_'+x for x in metrics]
    
    df2 = df[df['ind']==0].groupby(['bookingID'])[cols[0]].agg({'max','mean'}).reset_index()
    df2.columns = ['bookingID'] + ['speed_calib_'+x for x in metrics]
    
    df = pd.merge(df1, df2, on=['bookingID'], how="left")
    
    return df

def window_grp_bearing_06(df, cols = ['acc_cal_std'], stop_thres=8, thres = 1,turn_thres=300,
                   var_nm = "acc", metrics = {"sum","mean"}, sequence_metric=False):
    """
    Agg. features at sliding window of 10s and overlap window of 5s
    :cols = Agg. Columns
    :thres: threshold where we consider window as Acceleration event 
    :param x:window_feas 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    
    metrics = sorted(metrics)
    df['stop_ind'] = np.where(df["stop_sum"]>stop_thres,1,0)
    df['ind'] = df['missing_ind_max'] + df['signal_weak_max'] + df['stop_ind'] 
    df['ind1'] = np.where((df[cols[0]]>thres) & (df[cols[1]]-df[cols[2]] <turn_thres),1,0)
    
    df1 = df.groupby(['bookingID'])["ind1"].agg(metrics).reset_index()
    df1.columns = ['bookingID'] + [var_nm +"_" + x for x in df1.columns[1:]]
    
    df2 = df[df['ind']==0].groupby(['bookingID'])["ind1"].agg(metrics).reset_index()
    df2.columns = ['bookingID'] + [var_nm +"_calib_" + x for x in df2.columns[1:]]
    
   
    
    res = pd.merge(df1, df2, on=['bookingID'], how="left")
    return res

def trip_ending_fes_07(df):
    """
    Agg. features at sliding window of 10s and overlap window of 5s; Trip ending features
    :param x:window_feas 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
    
    df['speed_limit'] = np.where(df.Speed_mean>18,1,0)
    df = df.groupby(['bookingID'])["stop_max","speed_limit","missing_ind_max"].agg(get_length_sequences_last_00).reset_index()
    df.columns = ['bookingID','stop_max_trip_end','high_speed_dur_trip_end',"missing_trip_end"]
    return df

def window_grp_accuracy_08(df, cols = ['Accuracy_mean'], thres=8, metrics = {'max','mean',"min","std"}):
   
    metrics = sorted(metrics)
    df['stop_ind'] = np.where(df["stop_sum"]>thres,1,0)
    df['ind'] = df['missing_ind_max'] + df['signal_weak_max'] + df['stop_ind'] 
    df1 = df.groupby(['bookingID'])[cols[0]].agg(metrics).reset_index()
    df1.columns = ['bookingID'] + ["accuracy_"+x for x in metrics]
    
    df2 = df[df['ind']==0].groupby(['bookingID'])[cols[0]].agg(metrics).reset_index()
    df2.columns = ['bookingID'] + ["accuracy_calib_"+x for x in metrics]
    
    df = pd.merge(df1, df2, on=['bookingID'], how="left")
    
    return df

def generic_stats_features_09(df,cols= ["Speed","acc_cal","Accuracy"], thres=8):
    
    mean_cols =  [x+"_mean" for x in cols]
    df['stop_ind'] = np.where(df["stop_sum"]>thres,1,0)
    df['ind'] = df['missing_ind_max'] + df['signal_weak_max'] + df['stop_ind'] 
    df = df[df.ind==0]
    df = df.groupby(['bookingID'])[mean_cols].agg(quantile_00).reset_index()
    df.columns = ['bookingID'] + [x+"_q90" for x in cols]
    return df

def events_calculation_10(df, thres={"b":-0.5,"b_harsh":-3,"b_harsh_avg":-2,
                                  "a":0.5,"a_harsh":3,"a_harsh_avg":2}):
    
    """
    Agg. features at sliding window of 10s and overlap window of 5s
    :cols = Agg. Columns
    :thres: threshold where we consider window Detection event 
    :param x:window_feas 
    :type x: pandas.DataFrame
    :return type: DataFrame
    """
        
    
    df['stop_ind'] = np.where(df["stop_sum"]>8,1,0)
    df['unrel'] = df['missing_ind_max'] + df['signal_weak_max'] +  df['stop_ind']
    
    df['braking_event'] = np.where((df['acc_cal_mean']<thres['b']) | (df['acc_cal_min']<thres['b_harsh']),1,0)
    df['braking_harsh_event'] = np.where((df['acc_cal_mean']<thres['b_harsh_avg']) | (df['acc_cal_min']<thres['b_harsh']),1,0)
    df['braking_calib_event'] = np.where( df['unrel']==0,df['braking_event'],np.nan)
    df['braking_harsh_calib_event'] = np.where(df['unrel']==0,df['braking_harsh_event'],np.nan)
    
    
    df['acceleration_event'] = np.where((df['acc_cal_mean']>thres['a']) | (df['acc_cal_max']>thres['a_harsh']) ,1,0)
    df['acceleration_harsh_event'] = np.where((df['acc_cal_mean']>thres['a_harsh_avg']) | (df['acc_cal_max']>thres['a_harsh']),1,0)
    df['acceleration_calib_event'] = np.where( df['unrel']==0,df['acceleration_event'],np.nan)
    df['acceleration_harsh_calib_event'] = np.where(df['unrel']==0,df['acceleration_harsh_event'],np.nan)
    
    df['Speeding_event'] = np.where((df['Speed_mean']>20),1,0)
    df['Speeding_calib_event'] = np.where(df['unrel']==0,df['Speeding_event'],np.nan)
    
    
    
    event_cols = ["braking_event","braking_harsh_event","braking_calib_event","braking_harsh_calib_event",
            "acceleration_event","acceleration_harsh_event","acceleration_calib_event","acceleration_harsh_calib_event",
                 "Speeding_event"]
    
    df = df.groupby(['bookingID'])[event_cols].agg({"mean"}).reset_index()
    cols = ["bookingID"]
    for c in event_cols:
        cols.append(c) 
    df.columns = cols
    df["braking_harsh_ratio"] = df['braking_harsh_event']/df['braking_event']
    df["braking_harsh_calib_ratio"] = df['braking_harsh_calib_event']/df['braking_calib_event']
    
    df["acceleration_harsh_ratio"] = df['acceleration_harsh_event']/df['acceleration_event']
    df["acceleration_harsh_calib_ratio"] = df['acceleration_harsh_calib_event']/df['acceleration_calib_event']
    
    return df

def events_calculation_lstm_11(df, thres={"acceleration":1,"acc_cal":1,"Bearing":20,
                                  "Accuracy":3,"a_harsh":3,"a_harsh_avg":2}):
    
  
    df['acceleration_ind']  = np.where(df['acceleration_std']>thres['acceleration'],1,0)
    df['acc_cal_ind']  = np.where(df['acc_cal_std']>thres['acc_cal'],1,0)
    df['Bearing_ind']  = np.where(df['Bearing_std']>thres['Bearing'],1,0)
    df['Accuracy_ind']  = np.where(df['Accuracy_std']>thres['Accuracy'],1,0)
    
    
    
    
    event_cols = ['acceleration_ind','acc_cal_ind','Bearing_ind','Accuracy_ind']
    event_cols+=['stop_max','signal_weak_max','missing_ind_max']
    df = df.groupby(['bookingID','window'])[event_cols].max().reset_index()
    
    return df

def lstm_features_12(df, labels,train_ind=False):
    
    """
    LSTM features creation 
    """
        
    event_cols = [ 'acc_cal_ind','Bearing_ind','Accuracy_ind','signal_weak_max',
                "missing_ind_max"]
    sliding_feas = events_calculation_lstm_11(df)
    sliding_feas['event'] = sliding_feas[event_cols].sum(axis=1)
    events  = sliding_feas[sliding_feas.event==1]
    
    
    ## Keeping only one streak of zero event and stop event 
    zero_event = sliding_feas[sliding_feas['event']==0]
    zero_event['window_lag1'] = zero_event['window'].shift(+1)
    zero_event['streak'] = zero_event['window'] - zero_event['window_lag1']
    zero_event = zero_event[zero_event.streak!=1]
    
    stop_event = sliding_feas[sliding_feas['stop_max']==0]
    stop_event['window_lag1'] = stop_event['window'].shift(+1)
    stop_event['streak'] = stop_event['window'] - stop_event['window_lag1']
    stop_event = stop_event[stop_event.streak!=1]
    
    use_cols = [ 'acc_cal_ind','Bearing_ind','Accuracy_ind','stop_max','signal_weak_max','missing_ind_max']
    all_events = pd.concat([events, stop_event, zero_event], axis=0)
    all_events = all_events.groupby(['bookingID',"window"])[use_cols].max().reset_index()
    all_events['latest_point'] = all_events.groupby(['bookingID'])['window'].rank(ascending=False)
    all_events = all_events[all_events.latest_point<=100]
    
    
    if (train_ind):
        labels = labels.groupby(['bookingID']).max().reset_index()
        target = pd.Series(labels.label.values,index=labels.bookingID.values).to_dict()
    
    no_feas= len(use_cols)
    bids = all_events.bookingID.unique()
    no_bids = len(bids)
    X = []; Y=[]
    for i in range(no_bids):
        bid = bids[i]
        x = np.array(all_events[all_events.bookingID==bid][use_cols]); x_len = len(x)
        x.reshape(1,-1)
        X.append(x.reshape(len(x), no_feas))
        if (train_ind):
            Y.append(target[bid])
            
    X = np.array(X)
    X = pad_sequences(X,maxlen=100)
    if (train_ind):
        Y = np.array(Y)
        
    return X,Y,bids