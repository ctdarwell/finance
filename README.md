# finance
ML stock prediction

1. getSandP.py - download CSV of each stock's performance for 10 years
2. mkSP500_Close.py - organise SP500 CSVs into single CSV with single feature of Closing Value
3. sp500.py - use classifiers on features generated from returns AND identify subset of best performing stocks; considering up to 5 days lagged values; using the first and second moment of the historical log returns; randomly splits test/train data to guard against overfitting (not yet implemented cross-validated tho)
3. sp500.py - use classifiers on features generated from returns; considereing up to 5 days lagged values; using the first and second moment of the historical log returns; randomly splits test/train data to guard against overfitting (not yet implemented cross-validated tho)
4. finance.py - full Ch15 of Yves Hilpisch PDF for ML on finance

