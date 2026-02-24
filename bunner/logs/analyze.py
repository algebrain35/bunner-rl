import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def project_reward(df, future_t):
    projections = []

    last_row = df.iloc[-1]
    curr_reward = last_row['MeanReward']

    window=10
    reward_diff = df['MeanReward'].diff(window)
    delta_time = df['Minutes'].diff(window)
    reward_roc = reward_diff / delta_time
    smoothed_roc = reward_roc.rolling(window=8).mean()

    roc = smoothed_roc.iloc[-1]
    print("ROC: " + str(roc))
    decay = (smoothed_roc /smoothed_roc.shift(1)).rolling(window=8).median()
    retention = decay.iloc[-1]
    avg_episode_len = np.mean(df['TimeDelta'] / 60)

    curr_t = last_row['Minutes'] 
    proj_ts = np.arange(curr_t, curr_t + future_t + 1, 1)

    acc_gain = 0
    curr_iter_roc =roc

    for t in proj_ts:
        curr_iter_roc *= (retention ** (1 / avg_episode_len))
        acc_gain += curr_iter_roc
        projections.append(curr_reward + acc_gain)
    return pd.DataFrame({'Future Times': proj_ts, 'ProjReward': projections})



train_logs = pd.read_csv("./log", sep='\s+')
train_logs['Time'] = pd.to_datetime(train_logs['Time'])
train_logs['Minutes'] = (train_logs['Time'] - train_logs['Time'].iloc[0]).dt.total_seconds() / 60.0
rewards = train_logs['MeanReward']
reward_diff = train_logs['MeanReward'] - train_logs['MeanReward'].shift(1)

reward_diff.dropna(inplace=True)

dt = train_logs['TimeDelta'].iloc[1:]
reward_roc = reward_diff / dt

episodes = train_logs['Episode'].iloc[1:]
smoothed_roc = reward_roc.rolling(window=10).mean()

def avg_batch_time(df):
    times = df["Minutes"].diff()
    
    return np.mean(times)

print(avg_batch_time(train_logs))
if sys.argv[1] == "roc":
    try:
        ts = train_logs['Minutes'].iloc[1:]
        plt.plot(ts, reward_roc, alpha=0.3, label="Raw-RoC")
        plt.plot(ts, smoothed_roc, color="red", label="Trend")
        plt.axhline(0, color='black', linestyle="--")

    except Exception as e:
        print(np.shape(reward_roc))
        print(np.shape(smoothed_roc))
        print(np.shape(ts))
        print(e)
elif sys.argv[1] == "proj":
    ts = train_logs['Minutes']

    proj_df = project_reward(train_logs, 500*avg_batch_time(train_logs))
    print(len(proj_df['ProjReward']))
    plt.plot(proj_df["Future Times"], proj_df['ProjReward'], color="green", label="Decay")
    plt.plot(ts, rewards)


#reward_stability = train_logs['MeanReward'].rolling(window=5).var()
#plt.plot(train_logs['Minutes'], np.log(reward_stability))
plt.show()
