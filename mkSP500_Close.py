#build DF of Closing prices only for S&P500 full data files
import pandas as pd, sys, glob

df = pd.DataFrame()
files = glob.glob1('SP500', '*_data.csv') #data in folder - SP500


for file in files:
    col = file.split('_')[0]
    dat = pd.DataFrame(pd.read_csv(f"SP500/{file}")[['Date', 'Close']])
    dat.columns = ['Date', col]
    try: df = pd.merge(df, dat, on='Date', how='outer')
    except: df = dat.copy()

df.to_csv('SP500_Close.csv', index=False)

