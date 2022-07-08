#Modified from Yves Hilpisch - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch15/15_trading_strategies_b.ipynb

import numpy as np, sys
import pandas as pd
from pylab import mpl, plt
import requests
import io
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

#fnc to create lag columns of returns
def create_lags(data):
    global cols
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)


#Frequency Approach - one might transform the two real-valued features to
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


# Classification
#fit models based on the binary feature values and the derivation of the resulting position values
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

def fit_models(data): #fit all the models NB using the cols_bin global var from creat_bins func
    mfit = {model: models[model].fit(data[cols_bin], data['direction']) for model in models.keys()}

#make predictions
def derive_positions(data): #A function that derives all position values from the fitted models.
    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])


#Main
raw = pd.read_csv('SP500_Close.csv', index_col=0, parse_dates=True)
best_mods = []

#509 Sequential Train-Test Split
#now only GNB & LR give positive absolute performance (not sure this comment follows looking at the figure??)
from sklearn.model_selection import train_test_split

for symbol in raw.columns:
    data = pd.DataFrame(raw[symbol]).dropna()
    
    data['returns'] = np.log(data / data.shift(1))
    data['direction'] = np.sign(data['returns'])
    
    lags = 5  
    create_lags(data)
    data.dropna(inplace=True)

    #next code uses the first and second moment of the historical log
    #returns to digitize the features data, allowing for more possible feature value combinations.
    #This improves the performance of all classification algorithms used esp SVM again!
    mu = data['returns'].mean()  #digitise feats w mean & sd
    v = data['returns'].std()  
    bins = [mu - v, mu, mu + v]  #digitised here
    create_bins(data, bins)
    
    C = 1
    models = {'log_reg': linear_model.LogisticRegression(C=C),
              'gauss_nb': GaussianNB(),'svm': SVC(C=C)}

    #p511 Randomized Train-Test Split - mimic new patterns emerging
    train, test = train_test_split(data, test_size=0.5,
                                   shuffle=True, random_state=100)
    train = train.copy().sort_index()  
    train[cols_bin].head()
    
    test = test.copy().sort_index()  
    fit_models(train)
    derive_positions(test)
    evaluate_strats(test)

    #make df of strategy performance and choose best
    eval_tab = pd.DataFrame(test[sel].sum().apply(np.exp))
    chosen, val = eval_tab.sort_values(by = 0).index[-1], eval_tab.sort_values(by = 0).iloc[-1, -1]

    best_mods.append([symbol, chosen, val])


best_mods = pd.DataFrame(best_mods)
best_mods.columns = ['stock','strat','performance']

best_mods = best_mods.sort_values('performance', ascending=False)
print(best_mods.head())

best_mods.to_csv('SP500_performanceEval_rndmzSplit.csv', index=False)

# which strategies perform best most frequently in top 50
for strat in best_mods.strat.unique():
    print(strat, best_mods.iloc[:50, :].strat.tolist().count(strat))

