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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score
from pprint import pprint
from ta.trend import AroonIndicator
import warnings
warnings.filterwarnings("ignore")

def calculate_heikin_ashi(data):
    ha_data = data[['datetime', 'open_btc', 'high_btc', 'low_btc', 'close_btc']].copy()

    ha_data['HA_Close'] = (ha_data['open_btc'] + ha_data['high_btc'] + ha_data['low_btc'] + ha_data['close_btc']) / 4

    ha_data['HA_Open'] = 0.0  # Placeholder
    ha_data.iloc[0, ha_data.columns.get_loc('HA_Open')] = (ha_data.iloc[0]['open_btc'] + ha_data.iloc[0]['close_btc']) / 2

    for i in range(1, len(ha_data)):
        ha_data.iloc[i, ha_data.columns.get_loc('HA_Open')] = (
            ha_data.iloc[i - 1]['HA_Open'] + ha_data.iloc[i - 1]['HA_Close']
        ) / 2

    ha_data['HA_High'] = ha_data[['high_btc', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_data['HA_Low'] = ha_data[['low_btc', 'HA_Open', 'HA_Close']].min(axis=1)

    # Return the dataframe with Heikin-Ashi values
    ha_data = ha_data[['datetime', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    ha_data['HA_Highlow'] = ha_data['HA_High'] - ha_data['HA_Low']
    ha_data['HA_Highlow_square'] = ha_data['HA_Highlow']**2
    ha_data['log_HA_Close'] = np.log(ha_data['HA_Close'])
    ha_data['log_HA_Open'] = np.log(ha_data['HA_Open'])
    ha_data['log_HA_High'] = np.log(ha_data['HA_High'])
    ha_data['log_HA_Low'] = np.log(ha_data['HA_Low'])
    ha_data['log_HA_Highlow'] = np.log(ha_data['HA_Highlow'])
    ha_data['log_HA_Highlow_square'] = np.log(ha_data['HA_Highlow_square'])
    return ha_data['HA_Open'],ha_data['HA_High'],ha_data['HA_Low'],ha_data['HA_Close'], ha_data['HA_Highlow'], ha_data['HA_Highlow_square'], ha_data['log_HA_Close'], ha_data['log_HA_Open'], ha_data['log_HA_High'], ha_data['log_HA_Low'], ha_data['log_HA_Highlow'], ha_data['log_HA_Highlow_square']

def calculate_adx(data, coin, period=14):
    high, low, close = data[f'high_{coin}'], data[f'low_{coin}'], data[f'close_{coin}']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    data['ADX'] = adx
    return data['ADX']

def calculate_vwap(data, coin):
    typical_price = (data[f'high_{coin}'] + data[f'low_{coin}'] + data[f'close_{coin}']) / 3
    
    cumulative_tp_vol = (typical_price * data[f'volume_{coin}']).cumsum()
    cumulative_vol = data[f'volume_{coin}'].cumsum()
    
    vwap = cumulative_tp_vol / cumulative_vol
    
    data['VWAP'] = vwap
    return data['VWAP']

def compute_additional_features(data, coin, window_rsi=14, window_volatility=14, ma_window=50):
    data[f"Rolling_Volatility_{coin}"] = data[f"return_{coin}"].rolling(window=window_volatility).std()

    min_rsi = data[f"RSI_{coin}"].rolling(window=window_rsi).min()
    max_rsi = data[f"RSI_{coin}"].rolling(window=window_rsi).max()
    data[f"Stoch_RSI_{coin}"] = (data[f"RSI_{coin}"] - min_rsi) / (max_rsi - min_rsi)

    data[f"Momentum_{coin}"] = data[f"close_{coin}"].diff(periods=window_rsi)

    data[f"MA_{coin}"] = data[f"close_{coin}"].rolling(window=ma_window).mean()
    data[f"Distance_MA_{coin}"] = data[f"close_{coin}"] - data[f"MA_{coin}"]

    data[f"Lagged_Return_{coin}"] = data[f"return_{coin}"].shift(1)
    data[f"RSI_MACD_Ratio_{coin}"] = data[f"RSI_{coin}"] / (data[f"MACD_{coin}"] + 1e-9)

    data.drop(columns=[f"MA_{coin}"], inplace=True) 
    return data

def scale_features(data,columns_to_scale):
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data

def process_data(data):
    """
    Process the input data and return a dataframe with all the necessary indicators and data for making signals.

    Parameters:
    data (pandas.DataFrame): The input data to be processed.

    Returns:
    pandas.DataFrame: The processed dataframe with all the necessary indicators and data.
    """
    coin = "btc"
    data[f'open_{coin}'],data[f'high_{coin}'],data[f'low_{coin}'],data[f'close_{coin}'], data[f'high_low_{coin}'], data[f'high_low_square_{coin}'], data[f'log_close_{coin}'], data[f'log_open_{coin}'], data[f'log_high_{coin}'], data[f'log_low_{coin}'], data[f'log_high_low_{coin}'], data[f'log_high_low_square_{coin}'] = calculate_heikin_ashi(data)

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
    
    data[f'price_deviation_{coin}'] = data[f'close_{coin}'] - data[f'close_{coin}'].rolling(window=1000).mean()

    data[f'tr_high_low_{coin}'] = data[f'high_{coin}'] - data[f'low_{coin}']
    data[f'tr_high_close_{coin}'] = abs(data[f'high_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'tr_low_close_{coin}'] = abs(data[f'low_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'true_range_{coin}'] = data[[f'tr_high_low_{coin}', f'tr_high_close_{coin}', f'tr_low_close_{coin}']].max(axis=1)

    atr_period = 50
    data[f'ATR_{coin}'] = data[f'true_range_{coin}'].rolling(window=atr_period).mean()
    data[f'average_ATR_last_100_{coin}'] = data[f'ATR_{coin}'].rolling(window=100).mean()
    data[f'ATR_Ratio_{coin}']=data[f'ATR_{coin}']/data[f'average_ATR_last_100_{coin}']
    data[f'multiplier_{coin}'] = data[f'ATR_Ratio_{coin}'].apply(lambda x: 1 if x < 0.375 else (3 if 0.75 <= x < 1.25 else 2))
    data[f'position_signal_{coin}'] = data[f'multiplier_{coin}'].map({1: -1, 2: 0, 3: 1})

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

    data["chikou_span_btc"] = data['close_btc'].shift(26)
    data["MOM_btc"] = data["close_btc"]-data["chikou_span_btc"]

    data["chikou_span_eth"] = data['close_eth'].shift(26)
    data["MOM_eth"] = data["close_eth"]-data["chikou_span_eth"]

    window = 14 
    data['Lowest_Low_btc'] = data['low_btc'].rolling(window=window).min()
    data['Highest_High_btc'] = data['high_btc'].rolling(window=window).max()
    
    data['Fast_%K_btc'] = 100 * (data['close_btc'] - data['Lowest_Low_btc']) / (data['Highest_High_btc'] - data['Lowest_Low_btc'])
    data['Fast_%D_btc'] = data['Fast_%K_btc'].rolling(window=3).mean()
    data['Slow_%K_btc'] = data['Fast_%D_btc']
    data['Slow_%D_btc'] = data['Slow_%K_btc'].rolling(window=3).mean()

    data['Fast_Crossover_btc'] = (data['Fast_%K_btc'] > data['Fast_%D_btc']).astype(int) - (data['Fast_%K_btc'] < data['Fast_%D_btc']).astype(int)
    data['Slow_Crossover_btc'] = (data['Slow_%K_btc'] > data['Slow_%D_btc']).astype(int) - (data['Slow_%K_btc'] < data['Slow_%D_btc']).astype(int)

    data['Fast_Normalized_Distance_btc'] = (data['Fast_%K_btc'] - data['Fast_%D_btc']) / (data['Highest_High_btc'] - data['Lowest_Low_btc'])
    data['Slow_Normalized_Distance_btc'] = (data['Slow_%K_btc'] - data['Slow_%D_btc']) / (data['Highest_High_btc'] - data['Lowest_Low_btc'])

    data['Lowest_Low_eth'] = data['low_eth'].rolling(window=window).min()
    data['Highest_High_eth'] = data['high_eth'].rolling(window=window).max()

    data['Fast_%K_eth'] = 100 * (data['close_eth'] - data['Lowest_Low_eth']) / (data['Highest_High_eth'] - data['Lowest_Low_eth'])
    data['Fast_%D_eth'] = data['Fast_%K_eth'].rolling(window=3).mean()
    data['Slow_%K_eth'] = data['Fast_%D_eth']
    data['Slow_%D_eth'] = data['Slow_%K_eth'].rolling(window=3).mean()

    data['Fast_Crossover_eth'] = (data['Fast_%K_eth'] > data['Fast_%D_eth']).astype(int) - (data['Fast_%K_eth'] < data['Fast_%D_eth']).astype(int)
    data['Slow_Crossover_eth'] = (data['Slow_%K_eth'] > data['Slow_%D_eth']).astype(int) - (data['Slow_%K_eth'] < data['Slow_%D_eth']).astype(int)

    data['Fast_Normalized_Distance_eth'] = (data['Fast_%K_eth'] - data['Fast_%D_eth']) / (data['Highest_High_eth'] - data['Lowest_Low_eth'])
    data['Slow_Normalized_Distance_eth'] = (data['Slow_%K_eth'] - data['Slow_%D_eth']) / (data['Highest_High_eth'] - data['Lowest_Low_eth'])

    data[f'ema_btc'] = ta.trend.EMAIndicator(close=data['close_btc'], window=50).ema_indicator()
    data[f'ema_eth'] = ta.trend.EMAIndicator(close=data['close_eth'], window=50).ema_indicator()

    data[f'ema_percent_diff_{coin}'] = ((data[f'close_{coin}'] - data[f'ema_{coin}']) / data[f'ema_{coin}']) * 100

    data[f'adx_{coin}'] = calculate_adx(data, coin, period=14)
    data[f"VWAP_{coin}"] = calculate_vwap(data, coin)

    tau = 50  
    data['MFM_btc'] = ((data['close_btc'] - data['low_btc']) - (data['high_btc'] - data['close_btc'])) / (data['high_btc'] - data['low_btc'])
    data['MFM_btc'].replace([np.inf, -np.inf], 0, inplace=True) 
    data['MFV_btc'] = data['MFM_btc'] * data['volume_btc']

    data['CMF_btc'] = data['MFV_btc'].rolling(window=tau).sum() / data['volume_btc'].rolling(window=tau).sum()
    data.drop(['MFM_btc', 'MFV_btc'], axis=1, inplace=True)

    data['MFM_eth'] = ((data['close_eth'] - data['low_eth']) - (data['high_eth'] - data['close_eth'])) / (data['high_eth'] - data['low_eth'])
    data['MFM_eth'].replace([np.inf, -np.inf], 0, inplace=True)
    data['MFV_eth'] = data['MFM_eth'] * data['volume_eth']

    data['CMF_eth'] = data['MFV_eth'].rolling(window=tau).sum() / data['volume_eth'].rolling(window=tau).sum()
    data.drop(['MFM_eth', 'MFV_eth'], axis=1, inplace=True)

    data['williams_eth'] = -100*((data['high_eth'].rolling(26).max())-(data['close_eth']))/((data['high_eth'].rolling(26).max())-(data['low_eth'].rolling(14).min()))
    data['williams_btc'] = -100*((data['high_btc'].rolling(26).max())-(data['close_btc']))/((data['high_btc'].rolling(26).max())-(data['low_btc'].rolling(14).min()))

    data['OBV_btc'] = 0  
    data['OBV_btc'][0] = data['volume_btc'][0] 
    for i in range(1, len(data)):
        if data['close_btc'][i] > data['close_btc'][i-1]:
            data['OBV_btc'][i] = data['OBV_btc'][i-1] + data['volume_btc'][i] 
        elif data['close_btc'][i] < data['close_btc'][i-1]:
            data['OBV_btc'][i] = data['OBV_btc'][i-1] - data['volume_btc'][i]  
        else:
            data['OBV_btc'][i] = data['OBV_btc'][i-1]  

    data['OBV_eth'] = 0
    data['OBV_eth'][0] = data['volume_eth'][0]
    for i in range(1, len(data)):
        if data['close_eth'][i] > data['close_eth'][i-1]:
            data['OBV_eth'][i] = data['OBV_eth'][i-1] + data['volume_eth'][i]
        elif data['close_eth'][i] < data['close_eth'][i-1]:
            data['OBV_eth'][i] = data['OBV_eth'][i-1] - data['volume_eth'][i]
        else:
            data['OBV_eth'][i] = data['OBV_eth'][i-1]

    mx = 0 
    mn = 1000000000000
    data['run_max_btc'] = 0
    data['run_min_btc'] = 0
    for i in range(len(data)):
        mx =max(mx, data['close_btc'][i])
        mn =min(mn, data['close_btc'][i])
        data['run_max_btc'][i] = mx 
        data['run_min_btc'][i] = mn

    band_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    data['diff_btc'] = data['run_max_btc'] - data['run_min_btc']
    for level in band_levels:
        data['band_btc_'+str(level)] = data['run_max_btc'] - data['diff_btc']*level
    
    data['direction_btc'] = 0 
    for i in range(100,len(data)):
        data['direction_btc'][i] = (data['close_btc'][i] - data['close_btc'][i-100])/abs(data['close_btc'][i] - data['close_btc'][i-100])

    def close_enuff(x, y):
        return x/y > 0.99 and x/y < 1.01

    data['signals_btc'] = 0 
    for i in range(100,len(data)):
        for level in band_levels:
            if close_enuff(data['close_btc'][i], data['band_btc_'+str(level)][i]) and ((data['run_max_btc'][i] / data['run_min_btc'][i]) > 4):
                data['signals_btc'][i]  = -data['direction_btc'][i] 

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

    transformed_signals,tradess = transform(data['signals_btc'])
    data['signals2_btc'] = transformed_signals 

    mx = 0
    mn = 1000000000000
    data['run_max_eth'] = 0
    data['run_min_eth'] = 0
    for i in range(len(data)):
        mx = max(mx, data['close_eth'][i])
        mn = min(mn, data['close_eth'][i])
        data['run_max_eth'][i] = mx
        data['run_min_eth'][i] = mn

    band_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    data['diff_eth'] = data['run_max_eth'] - data['run_min_eth']
    for level in band_levels:
        data['band_eth_'+str(level)] = data['run_max_eth'] - data['diff_eth']*level

    data['direction_eth'] = 0
    for i in range(100,len(data)):
        data['direction_eth'][i] = (data['close_eth'][i] - data['close_eth'][i-100])/abs(data['close_eth'][i] - data['close_eth'][i-100])

    data['signals_eth'] = 0
    for i in range(100,len(data)):
        for level in band_levels:
            if close_enuff(data['close_eth'][i], data['band_eth_'+str(level)][i]) and ((data['run_max_eth'][i] / data['run_min_eth'][i]) > 4):
                data['signals_eth'][i]  = -data['direction_eth'][i]

    transformed_signals,tradess = transform(data['signals_eth'])
    data['signals2_eth'] = transformed_signals

    aroon_indicator = AroonIndicator(high=data[f'high_{coin}'], low=data[f'low_{coin}'], window=window)
    data[f'Aroon_Up_{coin}'] = aroon_indicator.aroon_up()
    data[f'Aroon_Down_{coin}'] = aroon_indicator.aroon_down()

    data.dropna(inplace=True)  
    data = compute_additional_features(data, coin)
    columns_to_scale = [f"ATR_{coin}", f"ATR_Ratio_{coin}", f"ema_{coin}", f"adx_{coin}", f"VWAP_{coin}", f"Rolling_Volatility_{coin}", f"Stoch_RSI_{coin}", f"Momentum_{coin}", f"Distance_MA_{coin}", f"Lagged_Return_{coin}", f"RSI_MACD_Ratio_{coin}", f"CMF_{coin}", f"williams_{coin}"]
    data = scale_features(data, columns_to_scale)
    
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
    print("Hidden States Generated")
    return test_data

def or_clutch(sig1,sig2):
    a = []
    for i in range(len(sig1)):
        x = min(max(sig1.iloc[i]+sig2.iloc[i],-2),2)
        b= sig1.iloc[i]+sig2.iloc[i]
        x = (sig1.iloc[i]+sig2.iloc[i]+1)//2
        if (sig1.iloc[i]==-2 or sig2.iloc[i]==-2):
            x=-2
        if (sig1.iloc[i]==2 or sig2.iloc[i]==2):
            x=2
        if (sig1.iloc[i]*sig2.iloc[i]==-4):
            x=0
        a.append(x)
    print(len(sig1[sig1==-2]))
    print(len(sig2[sig2==-2]))
    return a 

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
    
    train_data = data.copy()
    test_data = data.copy()
    coin = "btc"

    test_data = get_hidden_state(train_data, test_data, coin)
    # OBV bad for BTC

    # to remember for 23
    features1 = [
        # f"close_{coin}",
        # f"returns_square_{coin}",
        # f"return_{coin}",
        # f"high_low_{coin}",
        # f"high_low_square_{coin}",
        # f"chikou_span_{coin}",
        # f"RSI_{coin}",
        # f"z_score_{coin}",
        # f"signals2_{coin}",
        # f"CMF_{coin}",
        # f"williams_{coin}",
        # f"ATR_{coin}",
        # f"ATR_Ratio_{coin}",
        # f"multiplier_{coin}",
        # f"Fast_%K_{coin}",
        # f"Fast_%D_{coin}",
        # f"Fast_Normalized_Distance_{coin}",
        # f"adx_{coin}",

        f"open_{coin}",
        f"returns_square_{coin}",
        f"return_{coin}",
        f"high_low_square_{coin}",
        "log_return"
    ]
    # optimal that I had used for 24
    features2 = [
        f"close_{coin}",
        f"returns_square_{coin}",
        f"return_{coin}",
        f"high_low_{coin}",
        f"high_low_square_{coin}",
        f"chikou_span_{coin}",
        f"RSI_{coin}",

        f"signals2_{coin}",
        # f"williams_{coin}",
        # f"ATR_{coin}",
        # f"ATR_Ratio_{coin}",
        # f"multiplier_{coin}",
        f"Fast_%K_{coin}",
        # f"Fast_%D_{coin}",
        # f"Fast_Normalized_Distance_{coin}",

        f"adx_{coin}",
    ]
    # features = [
    #     f"log_close_{coin}",
    #     f"log_high_low_square_{coin}",
    #     f"log_high_low_{coin}",


    # ]

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
    target = f"shifted_return_{coin}"
    results = linear_regression_model(train_data, test_data, features1, target, regularization="ridge")
    ans["predictions1"] = results["predictions"]

    results = linear_regression_model(train_data, test_data, features2, target, regularization="ridge")
    ans["predictions2"] = results["predictions"]
    ans = pd.merge(
        ans, pd.DataFrame(results["returns_shifted"]), left_index=True, right_index=True
    )
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
        ["open", "high", "low", "close", "volume", "predictions1", "predictions2", "returns_shifted"]
    ]

    # minimum return threshold to generate signal
    t = 0.0002
    t2 = 1000

    ans["greater_thresh"] = ans["predictions1"] > t
    ans["lower_thresh"] = ans["predictions1"] < -t

    ans["volat_g_thresh"] = ans["predictions1"] > t2
    ans["volat_l_thresh"] = ans["predictions1"] < -t2

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

    ans["signals1"] = signals


    t = 0.0003
    t2 = 1000

    ans["greater_thresh"] = ans["predictions2"] > t
    ans["lower_thresh"] = ans["predictions2"] < -t

    ans["volat_g_thresh"] = ans["predictions2"] > t2
    ans["volat_l_thresh"] = ans["predictions2"] < -t2

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

    ans["signals2"] = signals

    ans["signals"] =or_clutch(ans["signals1"], ans["signals2"])
    ans_write = ans[
        ["datetime", "open", "high", "low", "close", "volume", "signals"]
    ].copy()
    ans_write["trade_type"] = "close"
    # ans_write['signals'] = transform(ans_write['signals'])
    print("features 1 : ", features1 , "feature 2 :",features2, "\nthreshold : ", t, "\ncoin : ", coin)

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
            else:
                ans_write.loc[index, "trade_type"] = "hold"
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
            else:
                ans_write.loc[index, "trade_type"] = "hold"

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

    # 
    # ravarp : Fit on the training data
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
    print("corr : ", np.corrcoef(y_test, y_pred)[0][1])
    print(f"R² Score: {r2:.4f}")

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
    print("Sent for Testing")
    backtest_result = perform_backtest_large_csv(path_from_relative_path(csv_file_path), leverage = 2)
    # backtest_result = perform_backtest(csv_file_path)
    # No need to use following code if you are using perform_backtest_large_csv
    print(backtest_result)
    for value in backtest_result:
        print(value)


if __name__ == "__main__":
    main()