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
     
def process_data(data, coin = 'btc'):

    window = 20
    data['extraordinary_volume_btc'] = data['volume_btc'] > data['volume_btc'].rolling(window=window).mean() + 3*data['volume_btc'].std()
    data['extraordinary_volume_btc'] = data['volume_btc'] > data['volume_btc'].rolling(window=window).mean() + 3*data['volume_btc'].std()
    data['mean_price_btc'] = data['close_btc'].rolling(window=window).mean()
    data['mean_price_btc'] = data['close_btc'].rolling(window=window).mean()
    data['price_deviatoin_btc'] = data['close_btc'] - data['mean_price_btc']
    data['price_deviatoin_btc'] = data['close_btc'] - data['mean_price_btc']
    data['z_score_btc'] = data['price_deviatoin_btc'] / data['price_deviatoin_btc'].rolling(window=window).std()
    data['z_score_btc'] = data['price_deviatoin_btc'] / data['price_deviatoin_btc'].rolling(window=window).std()
    data['returns_square_btc'] = data['return_btc']**2 
    data['returns_square_btc'] = data['return_btc']**2
    data['liquidity_btc'] = data['volume_btc'] * data['close_btc'] 
    data['liquidity_btc']+=1 
    data['log_liquid_btc'] = np.log(data['liquidity_btc'])
    data['liquidity_btc'] = data['volume_btc'] * data['close_btc']
    data['liquidity_btc']+=1
    data['log_liquid_btc'] = np.log(data['liquidity_btc'])

    data[f'tr_high_low_{coin}'] = data[f'high_{coin}'] - data[f'low_{coin}']
    data[f'tr_high_close_{coin}'] = abs(data[f'high_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'tr_low_close_{coin}'] = abs(data[f'low_{coin}'] - data[f'close_{coin}'].shift(1))
    data[f'true_range_{coin}'] = data[[f'tr_high_low_{coin}', f'tr_high_close_{coin}', f'tr_low_close_{coin}']].max(axis=1)
    
    data['HA_Open_btc'],data['HA_High_btc'],data['HA_Low_btc'],data['HA_Close_btc'], data['HA_Highlow_btc'], data['HA_Highlow_square_btc'], data['log_HA_Close_btc'], data['log_HA_Open_btc'], data['log_HA_High_btc'], data['log_HA_Low_btc'],data['log_HA_Highlow_btc'], data['log_HA_Highlow_square_btc'] = calculate_heikin_ashi(data)
    data[f'RSI_btc'] = ta.momentum.RSIIndicator(close=data[f'close_btc'], window=14).rsi()    
    data.dropna(inplace = True)    
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

def get_position(data, thres, coin = 'btc'):
    positions = []
    for _, row in data.iterrows():
        ret = row[f'shifted_return_{coin}']
        if ret > thres:
            positions.append(1)
        elif ret < -thres:
            positions.append(-1)
        else:
            positions.append(0)  
    
    data["positions"] = positions
    print("Optimal  Positons Calculated")
    return data

def get_onehot(data, categorical_features, features):
    for feature in categorical_features:
        print(feature)
        onehot = pd.get_dummies(data[feature], prefix=feature)
        data = pd.concat([data.drop(columns=[feature]), onehot], axis=1)
        features.extend(onehot.columns)
    
    return data, features

# -------STRATEGY LOGIC--------#
def strat(data):
    t_start = '2023'
    t_test_start = '2020'
    t_train_start = '2020'
    df_train = data.copy()
    df_test = data.copy()

    coin = 'btc'
    print(len(df_train))
    print(len(df_test))
    data = get_hidden_state(df_train, data, coin)      
    target = 'positions'
    coin = 'btc' 
    #*****************************************************
    features = [f'RSI_{coin}'] 
    #*****************************************************
    print(features)
    df_train = data.copy()
    df_test = data.copy()

    X_train = df_train[features]
    y_train = get_position(df_train, thres = 0.0001)[target] 
        
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)


    # from sklearn.tree import plot_tree
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(18, 10))  # Adjust the size of the figure as necessary
    # # Plot the decision tree with appropriate features and class names
    # plot_tree(
    #     clf, 
    #     feature_names = features, 
    #     filled=True
    # )
    # plt.show()
    data["dec_tree_btc"] = clf.predict(data[features])

    lr_features = [
        f"returns_square_{coin}",
        f"return_{coin}",
        f'dec_tree_{coin}',
        f'log_HA_Close_{coin}',
        f'log_HA_Open_{coin}',
        f'log_HA_High_{coin}',
        f'log_HA_Low_{coin}'
    ] 
    lr_target = f"shifted_return_{coin}"

    df_train = data.copy()
    df_test = data.copy()
    
    results = linear_regression_model(df_train, df_test, lr_features, lr_target)
    
    ans = df_test[[f'open_{coin}', f'high_{coin}', f'low_{coin}', f'close_{coin}', f'volume_{coin}', 'hidden_state']]
    ans = pd.merge(ans,pd.DataFrame(results['returns_shifted']), left_index=True, right_index=True)
    ans['predictions'] = results['y_pred']
    ans.rename(columns = {f'open_{coin}':'open', f'high_{coin}':'high', f'low_{coin}':'low', f'close_{coin}':'close', f'volume_{coin}':'volume',f'shifted_return_{coin}':'returns_shifted'}, inplace = True)
    ans = ans [['open', 'high', 'low', 'close', 'volume', 'predictions','returns_shifted' , 'hidden_state']]

    # minimum return threshold to generate signal
    t = 0.0005
    t2 = 0
    ans['greater_thresh'] = ans['predictions'] > t
    ans['lower_thresh'] = ans['predictions'] < -t

    ans['volat_g_thresh'] = ans['predictions'] > t2
    ans['volat_l_thresh'] = ans['predictions'] < -t2

    ans['datetime'] = df_test.loc[ans.index, 'datetime']

    invested = 0 
    signals = []

    for i in range(len(df_test)):
        sig = ans['greater_thresh'].iloc[i].astype(int) - ans['lower_thresh'].iloc[i].astype(int)
        sig2 = ans['volat_g_thresh'].iloc[i].astype(int) - ans['volat_l_thresh'].iloc[i].astype(int) 
        lat_var = ans["hidden_state"].iloc[i].astype(int)
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
    ans_write = ans[['datetime', 'open', 'high', 'low', 'close', 'volume', 'signals']].copy()
    ans_write["trade_type"] = "close"
    print(len(ans_write))
    return ans_write



def linear_regression_model( df_train, df_test,features, target, regularization='ridge', alpha=1.0):

    # Define X and y
    X_train = df_train[features]
    X_test = df_test[features]

    y_train = df_train[target]
    y_test = df_test[target]

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose the model based on regularization parameter
    if regularization == 'ridge':
        model = Ridge(alpha=alpha)
    elif regularization == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_y_pred = model.predict(X_train)
    r2 = r2_score(y_test, y_pred)
    # print("corr : ", np.corrcoef(y_test, y_pred)[0][1])
    print(f"Test R² Score: {r2:.4f}")
    r2_all = r2_score(y_train, train_y_pred) 
    print(f"Train R² Score: {r2_all:.4f}")
    results = {
        'coefficients': model.coef_,
        'returns_shifted' : y_test ,
        'y_pred' : y_pred
    }
    # X = df[features]
    # y_all = model.predict(X)
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
    freq = '1d'
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