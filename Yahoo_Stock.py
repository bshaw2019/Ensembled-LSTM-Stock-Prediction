# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:43:57 2019

@author: Brandon
"""

import requests
import pandas as pd
import arrow
import datetime
import matplotlib.pyplot as plt

def get_quote_data(symbol='SBIN.NS', data_range='1d', data_interval='1m'):
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    body = data['chart']['result'][0]    
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('UTC').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])    
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    df.dropna(inplace=True)     #removing NaN rows
    df.columns = ['OPEN', 'HIGH','LOW','CLOSE','VOLUME']    #Renaming columns in pandas
    
    return df

data = get_quote_data('ROKU', '5d', '1m')


print(data)

plt.figure()
plt.plot(data['OPEN'])
plt.show()