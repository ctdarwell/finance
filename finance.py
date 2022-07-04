#Yves Hilpisch - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch15/15_trading_strategies_b.ipynb

#UNDER DEVELOPMENT

#p484 Simple Moving Averages (SMAs)

import numpy as np, sys
import pandas as pd
from pylab import mpl, plt
import requests
import io
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

try: raw = pd.read_csv('tr_eikon_eod_data.csv', index_col=0, parse_dates=True)
except:
    url = 'https://raw.githubusercontent.com/yhilpisch/py4fi2nd/master/source/tr_eikon_eod_data.csv'
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    df.to_csv('tr_eikon_eod_data.csv', index=False)
    raw = pd.read_csv('tr_eikon_eod_data.csv', index_col=0, parse_dates=True)

raw.info()
symbol = 'AAPL.O'

data = pd.DataFrame(raw[symbol]).dropna()

#SMA values for two different rolling window sizes
SMA1 = 42
SMA2 = 252

data['SMA1'] = data[symbol].rolling(SMA1).mean() #Calculates the values for the shorter SMA.
data['SMA2'] = data[symbol].rolling(SMA2).mean() #Calculates the values for the longer SMA


data.plot(figsize=(10, 6))

#The trading rules are:
#• Go long (buy) (= +1) when the shorter SMA is above the longer SMA.
#• Go short (sell) (= -1) when the shorter SMA is below the longer SMA
#for a long only strategy one would use +1 for a long position and 0 for a neutral position

data.dropna(inplace=True)
data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
data.tail()

#plot times to change strategy
ax = data.plot(secondary_y='Position', figsize=(10, 6))
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

#Q is strategy better than simply continuously going long on Apple stock?
#p487 Vectorized Backtesting
#NB it appears that log returns simply offers a straightforward method to ascribe a neg or pos value to the day's performnce
#thus, then we subsequently use np.exp to get the actual values
data['Returns'] = np.log(data[symbol] / data[symbol].shift(1)) #Calculates the log returns of the Apple stock (i.e., the benchmark investment).
data['Strategy'] = data['Position'].shift(1) * data['Returns'] #Multiplies the position values, shifted by one day, by the log returns of the Apple stock; the shift is required to avoid a foresight bias

#The basic idea is that the algorithm can only set up a position in the Apple stock given today’s market data (e.g., just before the close). The position then earns tomorrow’s return.

data.round(4).head()
data.dropna(inplace=True)
np.exp(data[['Returns', 'Strategy']].sum()) #Sums up the log returns for the strategy and the benchmark investment and calculates the exponential value to arrive at the absolute performance

#Calculate the annualized volatility for the strategy and the benchmark investment.
data[['Returns', 'Strategy']].std() * SMA2 ** 0.5  #the strategy WINS (NB results slightly different to book as DF is different for some reason)

#view comparative performances
ax = data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
data['Position'].plot(ax=ax, secondary_y='Position', style='--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

#p489 Optimization
#check all SMA range values
sys.exit(2)
from itertools import product
sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)
results = pd.DataFrame()

for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace=True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace=True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace=True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results = results.append(pd.DataFrame({'SMA1': SMA1, 'SMA2': SMA2,
                       'MARKET': perf['Returns'], 'STRATEGY': perf['Strategy'],
                       'OUT': perf['Strategy'] - perf['Returns']},
                       index=[0]), ignore_index=True)
    
    
#Best:
results.sort_values('OUT', ascending=False).head(7)

#this result is heavily dependent on the data set used and is prone to overfitting
#i.e. should test on other data - p491


#p491 Random Walk Hypothesis (RWH)
#p492: 'As a consequence, "the best predictor for tomorrow’s price, in a least-squares sense, is today’s price if the RWH applies"'

#The basic idea is that the market prices from yesterday and four more days back can be used to predict today’s market price

#create df with new lagged columns by 1-5 days
symbol = '.SPX'
data = pd.DataFrame(raw[symbol])
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    data[col] = data[symbol].shift(lag)

data.dropna(inplace=True)

#p493 OLS regression is straightforward to implement. As the optimal
#regression parameters show (see figure), lag_1 indeed is the most important one in predicting
#the market price based on OLS regression

#p494 any algorithmic trading strategy must prove its worth by proving that the RWH does not apply in general

#p494 Linear OLS Regression
raw = pd.read_csv('tr_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()
symbol = "EUR="
data = pd.DataFrame(raw[symbol])
data['returns'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data['direction'] = np.sign(data['returns']).astype(int)

data['returns'].hist(bins=35, figsize=(10, 6))

lags = 2
def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)

create_lags(data)

#p495 The first feature (lag_1) represents the log returns of the financial time series lagged by one day. The second feature (lag_2) lags the log returns by two days
#Log returns — in contrast to prices — are STATIONARY in general, which often is a necessary
#condition for the application of statistical and ML algorithms

data.dropna(inplace=True)

data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', figsize=(10, 6), colorbar=True)
plt.axvline(0, c='r', ls='--')
plt.axhline(0, c='r', ls='--')


from sklearn.linear_model import LinearRegression
model = LinearRegression()
data['pos_ols_1'] = model.fit(data[cols], data['returns']).predict(data[cols]) #The regression is implemented on the log returns directly...
data['pos_ols_2'] = model.fit(data[cols], data['direction']).predict(data[cols]) #… and on the direction data which is of primary interest

data[['pos_ols_1', 'pos_ols_2']].head()

#The real-valued predictions are transformed to directional values (+1, -1).
data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)

#The two approaches yield different directional predictions in general.
data['pos_ols_1'].value_counts() #but pos_ols_1 on returns gives more pos
data['pos_ols_2'].value_counts()

#However, both lead to a relatively large number of trades over time
(data['pos_ols_1'].diff() != 0).sum()
(data['pos_ols_2'].diff() != 0).sum()

#p498 Under these assumptions, both regression-based strategies outperform the benchmark passive investment
#while only the strategy trained on the direction of the market (i.e. #2) shows a positive overall performance
data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)


#Now Show the number of correct and false predictions by the strategies
(data['direction'] == data['pos_ols_1']).value_counts()
(data['direction'] == data['pos_ols_2']).value_counts() # #2 better

data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(10, 6))

#p499 Clustering
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, random_state=0)
model.fit(data[cols])
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, 
       n_clusters=2, n_init=10, random_state=0, tol=0.0001, verbose=0)
data['pos_clus'] = model.predict(data[cols])
data['pos_clus'] = np.where(data['pos_clus'] == 1, -1, 1)

#NB next line NOT in book but subsequent figure is reversed without it
#MORE IMPORTANTLY - performance plot 11 lines down is bollox w/o it!!
data['pos_clus'] = data['pos_clus'] * -1 

data['pos_clus'].values
plt.figure(figsize=(10, 6))
plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1],c=data['pos_clus'], cmap='coolwarm');

#compare the returns
data['strat_clus'] = data['pos_clus'] * data['returns']
data[['returns', 'strat_clus']].sum().apply(np.exp)
(data['direction'] == data['pos_clus']).value_counts()
data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10, 6));


#p501 Frequency Approach - one might transform the two real-valued features to
#binary ones and assess the probability of an upward and a downward movement,
#respectively, from the historical observations of such movements, given the four possible
#combinations for the two binary features ((0, 0), (0, 1), (1, 0), (1, 1)).

def create_bins(data, bins=[0]):
    global cols_bin
    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=bins) #Digitizes the feature values given the bins parameter
        cols_bin.append(col_bin)
create_bins(data)

data[cols_bin + ['direction']].head() #Shows the digitized feature values and the label values
grouped = data.groupby(cols_bin + ['direction'])
grouped.size() #Shows the frequency of the possible movements conditional on the feature value combinations

res = grouped['direction'].size().unstack(fill_value=0) #Transforms the DataFrame object to have the frequencies in columns
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

res.style.apply(highlight_max, axis=1) #Highlights the highest-frequency value per feature value combination

data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1) #Translates the findings given the frequencies to a trading strategy
(data['direction'] == data['pos_freq']).value_counts()
data['strat_freq'] = data['pos_freq'] * data['returns']
data[['returns', 'strat_freq']].sum().apply(np.exp)
data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(figsize=(10, 6))



#504 Classification
#fit models based on the binary feature values and the derivation of the resulting position values
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

C = 1
models = {'log_reg': linear_model.LogisticRegression(C=C),
          'gauss_nb': GaussianNB(),'svm': SVC(C=C)}

def fit_models(data): #fit all the models NB using the cols_bin global var from creat_bins func
    mfit = {model: models[model].fit(data[cols_bin], data['direction'])
            for model in models.keys()}

fit_models(data)

#make predictions
def derive_positions(data): #A function that derives all position values from the fitted models.
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])

derive_positions(data)

#Next the vectorized backtesting of the resulting trading strategies. Figure 15-12 visualizes the performance over time
#calc performance for ea model
def evaluate_strats(data): #A function that evaluates all resulting trading strategies.
    global sel
    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']
        sel.append(col)
    sel.insert(0, 'returns')
evaluate_strats(data)
sel.insert(1, 'strat_freq') #we also eval the frequency approach model


data[sel].sum().apply(np.exp) #Some strategies might show the exact same performance.

data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6)) #Some strategies show the exact same performance

#p506 Five Binary Features
#the following code works with five binary lags instead of two. 
#In particular, the performance of the SVM-based strategy is significantly improved - while others drop
data = pd.DataFrame(raw[symbol])

data['returns'] = np.log(data / data.shift(1))
data['direction'] = np.sign(data['returns'])

lags = 5  
create_lags(data)
data.dropna(inplace=True)

create_bins(data)  
data[cols_bin].head()

data.dropna(inplace=True)
fit_models(data)
derive_positions(data) #make predictions
evaluate_strats(data) #calc performance for ea model
#sel.remove('strat_freq') #maybe needed if don't run code in order
data[sel].sum().apply(np.exp)

data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6));
# plt.savefig('strat_ml_5bins.png')

#p508 Five Digitized Features
#Finally, the following code uses the first and second moment of the historical log
#returns to digitize the features data, allowing for more possible feature value combinations.
#This improves the performance of all classification algorithms used esp SVM again!

mu = data['returns'].mean()  #digitise feats w mean & sd
v = data['returns'].std()  
bins = [mu - v, mu, mu + v]  #digitised here
create_bins(data, bins)
data[cols_bin].head()

fit_models(data)
derive_positions(data)
evaluate_strats(data)
data[sel].sum().apply(np.exp) #SVM even better!!
data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6));
#plt.savefig('strat_ml_5bins_1&2moments.png')

#509 Sequential Train-Test Split
#now only GNB & LR give positive absolute performance (not sure this comment follows looking at the figure??)

split = int(len(data) * 0.5)
train = data.iloc[:split].copy()  
fit_models(train)  
test = data.iloc[split:].copy()  
derive_positions(test)  
evaluate_strats(test)  
test[sel].sum().apply(np.exp)

test[sel].cumsum().apply(np.exp).plot(figsize=(10, 6));


#p511 Randomized Train-Test Split - mimic new patterns emerging
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.5,
                               shuffle=True, random_state=100)
train = train.copy().sort_index()  
train[cols_bin].head()

test = test.copy().sort_index()  
fit_models(train)
derive_positions(test)
evaluate_strats(test)
test[sel].sum().apply(np.exp)

test[sel].cumsum().apply(np.exp).plot(figsize=(10, 6)) #SVM best again

print("From standard ML methods - why are predictions basically the same numbers as returns????")


#p512 Deep Neural Network
#DNN with scikit-learn

#First, it is trained and tested on the whole data set, using the digitized features
#exceptional in-sample performance (overfits)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=2 * [250], random_state=1)

model.fit(data[cols_bin], data['direction'])

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=[250, 250], learning_rate='constant',
              learning_rate_init=0.001, max_iter=200, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

#or from github: MLPClassifier(alpha=1e-05, hidden_layer_sizes=[250, 250], random_state=1, solver='lbfgs')

data['pos_dnn_sk'] = model.predict(data[cols_bin])
data['strat_dnn_sk'] = data['pos_dnn_sk'] * data['returns']
data[['returns', 'strat_dnn_sk']].sum().apply(np.exp)


data[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize=(10, 6))

#To avoid overfitting of the DNN model, a randomized train-test split is applied next
train, test = train_test_split(data, test_size=0.5, random_state=100)
train = train.copy().sort_index()
test = test.copy().sort_index()

model = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=500,
                     hidden_layer_sizes=3 * [500], random_state=1)  #Increases the number of hidden layers and hidden units.

model.fit(train[cols_bin], train['direction']) #NB some convergence issue here

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=[500, 500, 500], learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

test['pos_dnn_sk'] = model.predict(test[cols_bin])
test['strat_dnn_sk'] = test['pos_dnn_sk'] * test['returns']
test[['returns', 'strat_dnn_sk']].sum().apply(np.exp)
test[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize=(10, 6))


sys.exit(2) #!!!!!!!!!!!!!!!!!!!!!!!!!!

#p515 DNNs with TensorFlow
#In-sample, the algorithm outperforms the passive benchmark investment and shows a considerable absolute return (see Figure 15-19), again hinting at overfitting:

#this from GitHub
#added from prev due to some cuda/tf issue:
import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA Corporation\\cudart64_100.dll")
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential


def create_model():
    np.random.seed(100)
    tf.random.set_seed(100)
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=lags))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

data_ = (data - data.mean()) / data.std()
data['direction_'] = np.where(data['direction'] == 1, 1, 0)

model = create_model()
model.fit(data_[cols], data['direction_'],
          epochs=50, verbose=False)

model.evaluate(data_[cols], data['direction_'])

pred = np.where(model.predict(data_[cols]) > 0.5, 1, 0) 
pred[:10].flatten()

data['pos_dnn_ke'] = np.where(pred > 0, 1, -1)  
data['strat_dnn_ke'] = data['pos_dnn_ke'] * data['returns']
data[['returns', 'strat_dnn_ke']].sum().apply(np.exp)
data[['returns', 'strat_dnn_ke']].cumsum().apply(np.exp).plot(figsize=(10, 6))


#The following code again implements a randomized train-test split to get a more realistic view of the performance of the DNN-based algorithmic trading strategy
#avoid overfit

mu, std = train.mean(), train.std()
train_ = (train - mu) / mu.std()
model = create_model()

train['direction_'] = np.where(train['direction'] > 0, 1, 0)
model.fit(train_[cols], train['direction_'],
          epochs=50, verbose=False)

test_ = (test - mu) / std
test['direction_'] = np.where(test['direction'] > 0, 1, 0)
model.evaluate(test_[cols], test['direction_'])

pred = np.where(model.predict(test_[cols]) > 0.5, 1, 0) 
pred[:10].flatten()

test['pos_dnn_ke'] = np.where(pred > 0, 1, -1)
test['strat_dnn_ke'] = test['pos_dnn_ke'] * test['returns']
test[['returns', 'strat_dnn_sk', 'strat_dnn_ke']].sum().apply(np.exp)
test[['returns', 'strat_dnn_sk', 'strat_dnn_ke']].cumsum().apply(np.exp).plot(figsize=(10, 6));


#all works but strat_dnn_ke doesn't perform as from GH code

#CODE FROM BOOK p515 seems to set some weird attrs re tf - basically some issue with tf set up on this machine

