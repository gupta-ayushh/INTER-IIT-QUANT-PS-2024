"""
All modules and functions required for back_test should be added in requirements.txt.
"""
import uuid
import pandas as pd
from untrade.client import Client
import ta
import matplotlib.pyplot as plt
from untrade.client import Client
from pprint import pprint
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter1d
# ALL your imports here
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")

def process_data(data):
    """
    Process the input data and return a dataframe with all the necessary indicators and data for making signals.

    Parameters:
    data (pandas.DataFrame): The input data to be processed.

    Returns:
    pandas.DataFrame: The processed dataframe with all the necessary indicators and data.
    """
    coin = "btc"
    data['extraordinary_volume_eth'] = data['volume_eth'] > data['volume_eth'].rolling(window=1000).mean() + 3*data['volume_eth'].std()
    data['extraordinary_volume_btc'] = data['volume_btc'] > data['volume_btc'].rolling(window=1000).mean() + 3*data['volume_btc'].std()
    data['mean_price_btc'] = data['close_btc'].rolling(window=1000).mean()
    data['mean_price_eth'] = data['close_eth'].rolling(window=1000).mean()
    data['price_deviatoin_btc'] = data['close_btc'] - data['mean_price_btc']
    data['price_deviatoin_eth'] = data['close_eth'] - data['mean_price_eth']
    data['z_score_btc'] = data['price_deviatoin_btc'] / data['price_deviatoin_btc'].rolling(window=1000).std()
    data['z_score_eth'] = data['price_deviatoin_eth'] / data['price_deviatoin_eth'].rolling(window=1000).std()
    data['returns_square_btc'] = data['return_btc']**2 
    data['returns_square_eth'] = data['return_eth']**2
    data['liquidity_btc'] = data['volume_btc'] * data['close_btc'] 
    data['liquidity_btc']+=1 
    data['log_liquid_btc'] = np.log(data['liquidity_btc'])
    data['liquidity_eth'] = data['volume_eth'] * data['close_eth']
    data['liquidity_eth']+=1
    data['log_liquid_eth'] = np.log(data['liquidity_eth'])
    data['high_low_eth'] = data['high_eth'] - data['low_eth']
    data['high_low_btc'] = data['high_btc'] - data['low_btc']
    data['high_low_square_eth'] = data['high_low_eth']**2
    data['high_low_square_btc'] = data['high_low_btc']**2

    data[f'tr_high_low_{coin}'] = data[f'high_{coin}'] - data[f'low_{coin}']
    data[f'tr_high_close_{coin}'] = abs(data[f'high_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'tr_low_close_{coin}'] = abs(data[f'low_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'true_range_{coin}'] = data[[f'tr_high_low_{coin}', f'tr_high_close_{coin}', f'tr_low_close_{coin}']].max(axis=1)

    atr_period = 50
    data[f'ATR_{coin}'] = data[f'true_range_{coin}'].rolling(window=atr_period).mean()
    data[f'average_ATR_last_100_{coin}'] = data[f'ATR_{coin}'].rolling(window=100).mean()
    data[f'ATR_Ratio_{coin}']=data[f'ATR_{coin}']/data[f'average_ATR_last_100_{coin}']
    data[f'multiplier_{coin}'] = data[f'ATR_Ratio_{coin}'].apply(lambda x: 1 if x < 0.375 else (3 if 0.75 <= x < 1.25 else 2))

    data[f'RSI_eth'] = ta.momentum.RSIIndicator(close=data[f'close_eth'], window=14).rsi()
    macd_eth = ta.trend.MACD(close=data[f'close_eth'])
    data[f'MACD_eth'] = macd_eth.macd()
    data[f'MACD_Signal_eth'] = macd_eth.macd_signal()
    data[f'MACD_Hist_eth'] = macd_eth.macd_diff()

    data['RSI_btc'] = ta.momentum.RSIIndicator(close=data['close_btc'], window=14).rsi()
    macd_btc = ta.trend.MACD(close=data['close_btc'])
    data['MACD_btc'] = macd_btc.macd()
    data['MACD_Signal_btc'] = macd_btc.macd_signal()
    data['MACD_Hist_btc'] = macd_btc.macd_diff()

    bollinger_eth = ta.volatility.BollingerBands(close=data['close_eth'], window=20, window_dev=2)
    data['BB_upper_eth'] = bollinger_eth.bollinger_hband()
    data['BB_lower_eth'] = bollinger_eth.bollinger_lband()
    data['BB_middle_eth'] = bollinger_eth.bollinger_mavg()
    data['BB_bandwidth_eth'] = bollinger_eth.bollinger_wband()
    data['BB_percent_eth'] = bollinger_eth.bollinger_pband()

    bollinger_btc = ta.volatility.BollingerBands(close=data['close_btc'], window=20, window_dev=2)
    data['BB_upper_btc'] = bollinger_btc.bollinger_hband()
    data['BB_lower_btc'] = bollinger_btc.bollinger_lband()
    data['BB_middle_btc'] = bollinger_btc.bollinger_mavg()
    data['BB_bandwidth_btc'] = bollinger_btc.bollinger_wband()
    data['BB_percent_btc'] = bollinger_btc.bollinger_pband()
    data['chikou_span_eth']=data['close_eth'].shift(26)
    data['chikou_span_btc']=data['close_btc'].shift(26)
    data['williams_eth'] = -100*((data['high_eth'].rolling(14).max())-(data['close_eth']))/((data['high_eth'].rolling(14).max())-(data['low_eth'].rolling(14).min()))
    data['williams_btc'] = -100*((data['high_btc'].rolling(14).max())-(data['close_btc']))/((data['high_btc'].rolling(14).max())-(data['low_btc'].rolling(14).min()))
    data["log_return"] = np.log(data[f"close_{coin}"] / data[f"close_{coin}"].shift(1))
    
    data['OBV'] = 0  
    data['OBV'][0] = data['volume_eth'][0] 
    for i in range(1, len(data)):
        if data['close_eth'][i] > data['close_eth'][i-1]:
            data['OBV'][i] = data['OBV'][i-1] + data['volume_eth'][i] 
        elif data['close_eth'][i] < data['close_eth'][i-1]:
            data['OBV'][i] = data['OBV'][i-1] - data['volume_eth'][i]  
        else:
            data['OBV'][i] = data['OBV'][i-1]
    
    mx = 0 
    mn = 1000000000000
    data['run_max'] = 0
    data['run_min'] = 0
    for i in range(len(data)):
        mx =max(mx, data[f'close_{coin}'][i])
        mn =min(mn, data[f'close_{coin}'][i])
        data['run_max'][i] = mx 
        data['run_min'][i] = mn

    band_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    data['diff'] = data['run_max'] - data['run_min']
    for level in band_levels:
        data['band_'+str(level)] = data['run_max'] - data['diff']*level
    
    data['direction'] = 0 
    for i in range(100,len(data)):
        data['direction'][i] = (data[f'close_{coin}'][i] - data[f'close_{coin}'][i-100])/abs(data[f'close_{coin}'][i] - data[f'close_{coin}'][i-100])

    def close_enuff(x, y):
        return x/y > 0.99 and x/y < 1.01

    data['signals'] = 0 
    for i in range(100,len(data)):
        for level in band_levels:
            if close_enuff(data[f'close_{coin}'][i], data['band_'+str(level)][i]) and ((data['run_max'][i] / data['run_min'][i]) > 4):
                data['signals'][i]  = -data['direction'][i] 

    def transform(signals):
        pos=0
        final_signals=[]
        trade_type = []
        for signal in signals:
            if signal==0:
                trade_type.append('hold')
                final_signals.append(0)
            elif signal==1:
                if pos==0 :
                    trade_type.append('long')
                    final_signals.append(1)
                    pos+=1
                elif pos==-1:
                    trade_type.append('long reversal')
                    final_signals.append(2)
                    pos+=2
                else:
                    trade_type.append('hold')
                    final_signals.append(0)
            
            elif signal==-1:
                if pos==0 :
                    trade_type.append('short')
                    final_signals.append(-1)
                    pos-=1
                elif pos==1:
                    trade_type.append('short reversal')
                    pos-=2 
                    final_signals.append(-2)
                else:
                    trade_type.append('hold')
                    final_signals.append(0)
            else:
                trade_type.append('hold')
                final_signals.append(0)
        return final_signals,trade_type

    transformed_signals,tradess = transform(data['signals'])
    data[f'signals2_{coin}'] = transformed_signals 

    
    data.dropna(inplace=True)  
    return data

def get_hidden_state(train_data, test_data, coin):
    test_data["log_return"] = np.log(test_data[f"close_{coin}"] / test_data[f"close_{coin}"].shift(1))
    test_data.dropna(inplace=True)  
    test_rets = test_data["log_return"].values.reshape(-1, 1)  

    train_data["log_return"] = np.log(train_data[f"close_{coin}"] / train_data[f"close_{coin}"].shift(1))
    train_data.dropna(inplace=True)  
    train_rets = train_data["log_return"].values.reshape(-1, 1)  
    
    n_states = 2  
    hmm_model = GaussianHMM(
        n_components=n_states,           
        covariance_type="full",
        random_state=42, 
        tol=1e-5)
    hmm_model.fit(train_rets)
    train_data["hidden_state"] = hmm_model.predict(train_rets)
    test_data["hidden_state"] = hmm_model.predict(test_rets)
    return test_data

# -------STRATEGY LOGIC--------#
def strat(data):
    """
    Create a strategy based on indicators or other factors.

    Parameters:
    - data: DataFrame
        The input data containing the necessary columns for strategy creation.

    Returns:
    - DataFrame
        The modified input data with an additional 'signal' column representing the strategy signals.
    """
    coin = "btc"
    test_data = data.copy()
    train_data = data.copy()
    test_data = get_hidden_state(train_data, test_data, coin)
    features = [
        f"open_{coin}",
        # f"close_{coin}",
        f"returns_square_{coin}",
        f"return_{coin}",
        # f"z_score_{coin}",
        # f"high_low_{coin}",
        f"high_low_square_{coin}",
        # f"RSI_{coin}",
        # f"chikou_span_{coin}",
        # f"signals2_{coin}",
        # f"price_deviatoin_{coin}"
        # "OBV",
        "log_return"
    ]

    target = f"shifted_return_{coin}"
    results = linear_regression_model(train_data, test_data, features, target, regularization="ridge")
    ans = test_data[
        [
            "open_btc",
            "high_btc",
            "low_btc",
            "close_btc",
            "volume_btc",
            "open_eth",
            "high_eth",
            "low_eth",
            "close_eth",
            "volume_eth",
        ]
    ]
    ans = pd.merge(
        ans, pd.DataFrame(results["returns_shifted"]), left_index=True, right_index=True
    )
    ans["predictions"] = results["predictions"]
    ans.rename(
        columns={
            f"open_{coin}": "open",
            f"high_{coin}": "high",
            f"low_{coin}": "low",
            f"close_{coin}": "close",
            f"volume_{coin}": "volume",
            f"shifted_return_{coin}": "returns_shifted",
        },
        inplace=True,
    )
    ans = ans[
        ["open", "high", "low", "close", "volume", "predictions", "returns_shifted"]
    ]

    # minimum return threshold to generate signal
    t = 0.00009
    t2 = 9999999999990.000
    with open("final_btc_1h.txt", "a") as file:
        file.write(f"threshold 2: {t2}\n")

    ans["greater_thresh"] = ans["predictions"] > t
    ans["lower_thresh"] = ans["predictions"] < -t

    ans["volat_g_thresh"] = ans["predictions"] > t2
    ans["volat_l_thresh"] = ans["predictions"] < -t2

    # ans['signals'] = ans['greater_thresh'].astype(int) - ans['lower_thresh'].astype(int)
    ans["datetime"] = test_data.loc[ans.index, "datetime"]

    invested = 0
    signals = []
    for i in range(len(test_data)):
        sig = ans["greater_thresh"].iloc[i].astype(int) - ans["lower_thresh"].iloc[
            i
        ].astype(int)
        sig2 = ans["volat_g_thresh"].iloc[i].astype(int) - ans["volat_l_thresh"].iloc[
            i
        ].astype(int)
        lat_var = test_data["hidden_state"].iloc[i].astype(int)
        if sig == 1 and invested != 1:
            if invested == 0 and lat_var == 0:
                signals.append(1)
                invested = 1
            elif invested == -1 and lat_var == 0:
                signals.append(2)
                invested = 1
            elif invested == -1 and lat_var == 1:
                if sig2 != 0:
                    signals.append(2)
                    invested = 1
                else:
                    signals.append(1)
                    invested = 0
            elif invested == 0 and lat_var == 1:
                signals.append(sig2)
                invested += sig2
        elif sig == -1 and invested != -1:
            if invested == 0 and lat_var == 0:
                signals.append(-1)
                invested = -1
            elif invested == 1 and lat_var == 0:
                signals.append(-2)
                invested = -1
            elif invested == 1 and lat_var == 1:
                if sig2 != 0:
                    signals.append(-2)
                    invested = -1
                else:
                    signals.append(-1)
                    invested = 0
            elif invested == 0 and lat_var == 1:
                signals.append(sig2)
                invested += sig2
        else:
            signals.append(0)

    ans["signals"] = signals
    ans_write = ans[
        ["datetime", "open", "high", "low", "close", "volume", "signals"]
    ].copy()
    ans_write["trade_type"] = "close"
    # ans_write['signals'] = transform(ans_write['signals'])
    with open("final_btc_1h.txt", "a") as file:
        file.write(f"features: {features}\nthreshold: {t}\ncoin: {coin}\n")


    pos = 0
    entry_price = 0
    for index, row in ans_write.iterrows():
        if pos == 0:
            if row["signals"] == 0:
                ans_write.loc[index, "trade_type"] = "hold"
            if row["signals"] == 1:
                ans_write.loc[index, "trade_type"] = "long"
                entry_price = row["close"]
                pos += 1
            elif row["signals"] == -1:
                ans_write.loc[index, "trade_type"] = "short"
                entry_price = row["close"]
                pos -= 1
            elif row["signals"] == 2:
                entry_price = row["close"]
                ans_write.loc[index, "trade_type"] = "long"
                pos += 1
            elif row["signals"] == -2:
                entry_price = row["close"]
                ans_write.loc[index, "trade_type"] = "short"
                pos -= 1
        elif pos == 1:
            if row["close"] <= entry_price * 0.65:
                ans_write.loc[index, "trade_type"] = "close"
                ans_write.loc[index, "signals"] = -1
                pos = 0
            elif row["signals"] == 0:
                ans_write.loc[index, "trade_type"] = "hold"
            elif row["signals"] == 1:
                ans_write.loc[index, "trade_type"] = "hold"
            elif row["signals"] == -1:
                ans_write.loc[index, "trade_type"] = "close"
                pos -= 1
            elif row["signals"] == -2:
                ans_write.loc[index, "trade_type"] = "short reversal"
                entry_price = row["close"]
                pos -= 2
        elif pos == -1:
            if row["close"] >= 1.35 * entry_price:

                ans_write.loc[index, "trade_type"] = "close"
                ans_write.loc[index, "signals"] = 1
                pos = 0
            elif row["signals"] == 0:
                ans_write.loc[index, "trade_type"] = "hold"
            elif row["signals"] == 1:
                ans_write.loc[index, "trade_type"] = "close"
                pos += 1
            elif row["signals"] == -1:
                ans_write.loc[index, "trade_type"] = "hold"
            elif row["signals"] == 2:
                entry_price = row["close"]
                ans_write.loc[index, "trade_type"] = "long reversal"
                pos += 2

    return ans_write


def transform(signals):
    pos = 0
    final_signals = []
    for signal in signals:
        if signal == 0:
            final_signals.append(0)
        elif signal == 1:
            if pos == 0:
                final_signals.append(1)
                pos += 1
            elif pos == -1:
                final_signals.append(2)
                pos += 2
            else:
                final_signals.append(0)

        elif signal == -1:
            if pos == 0:
                final_signals.append(-1)
                pos -= 1
            elif pos == 1:
                pos -= 2
                final_signals.append(-2)
            else:
                final_signals.append(0)
    return final_signals


def linear_regression_model(df_train, df_test, features, target, regularization="ridge", alpha=1.0):
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]    

    scaler = StandardScaler()

    # Fit on the training data
    X_train = scaler.fit_transform(X_train)

    # Transform test and validation data using the same scaler
    X_test = scaler.transform(X_test)

    # Choose the model based on regularization parameter
    if regularization == "ridge":
        model = Ridge(alpha=alpha)
    elif regularization == "lasso":
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    with open("final_btc_1h.txt", "a") as file :    
        file.write(f"corr : {np.corrcoef(y_test, y_pred)[0][1]}\n")
        file.write(f"R² Score: {r2:.4f}\n")


    results = {
        "coefficients": model.coef_,
        "r2_score": r2,
        "predictions": y_pred,          #! Important
        "returns_shifted": y_test,      #! Important
    }

    return results

def perform_backtest(csv_file_path, leverage):
    client = Client()
    result = client.backtest(
        jupyter_id="team32_zelta_hpps",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=leverage,  # Adjust leverage as needed
    )
    return result

# Following function can be used for every size of file, specially for large files(time consuming,depends on upload speed and file size)
def perform_backtest_large_csv(csv_file_path, leverage):
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        # Normal Backtest
        result = client.backtest(
            file_path=csv_file_path,
            leverage=leverage,
            jupyter_id="team32_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=leverage,
                jupyter_id="team32_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )

            for value in result:
                print(value)

            os.remove(chunk_file_path)

            chunk_number += 1

    return result

def path_from_relative_path(relative_path):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV file
    file_path = os.path.join(script_dir, relative_path)
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV file
    file_path = os.path.join(script_dir, relative_path)

    return file_path

def preprocess_and_save_data():
    freq = '1h'
    df_btc = pd.read_csv(path_from_relative_path(f'BTC_2019_2023_{freq}.csv'), index_col = 'datetime')
    df_eth = pd.read_csv(path_from_relative_path(f'ETHUSDT_{freq}.csv'), index_col = 'datetime')

    df_btc.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_eth.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_btc.index = pd.to_datetime(df_btc.index)
    df_eth.index = pd.to_datetime(df_eth.index)

    df_btc = df_btc.loc['2020':'2023']
    df_eth = df_eth.loc['2020':'2023']

    df_btce = pd.merge(df_btc, df_eth, how = 'inner', left_index = True, right_index = True, suffixes = ('_btc', '_eth'))

    df_btce['return_btc'] = df_btce['close_btc'].pct_change()
    df_btce['return_eth'] = df_btce['close_eth'].pct_change()

    # shift the returns
    df_btce['shifted_return_btc'] = df_btce['return_btc'].shift(-1)
    df_btce['shifted_return_eth'] = df_btce['return_eth'].shift(-1)

    return df_btce

def main():
    data = preprocess_and_save_data()
    data.reset_index(inplace=True)

    processed_data = process_data(data)

    result_data = strat(processed_data)

    csv_file_path = "results.csv"
    result_data.to_csv(path_from_relative_path(csv_file_path), index=False)
    # print("Sent for Testing")
    backtest_result = perform_backtest_large_csv(path_from_relative_path(csv_file_path), leverage = 3)
    # backtest_result = perform_backtest(csv_file_path)
    # No need to use following code if you are using perform_backtest_large_csv
    print(backtest_result)
    for value in backtest_result:
        print(value)


if __name__ == "__main__":
    main()