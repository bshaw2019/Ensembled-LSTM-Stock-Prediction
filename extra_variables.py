# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 20:04:47 2018

@author: Brandon
"""

import numpy as np

def addVariables(df, ticker):
    df = df[::-1]
    print('Calculating MACD')
    
    group = df['Close']
    
    long = group.ewm(span=26,adjust=False).mean()
    short = group.ewm(span=12,adjust=False).mean()
    macd = short-long
    signal = macd.ewm(span=9,adjust=False).mean()
    
    
    df['MACD'] = macd
    df['signal_line'] = signal
    MACD_signals = []
    
    for i in range(len(df)):
        
        if i <= 1:
            MACD_signals.append(0)
        
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            previous_two = df.iloc[i-2,:]
            
        
            if day['MACD'] > previous['MACD'] and previous['MACD'] < previous_two['MACD']:
                MACD_signals.append(1)
            else:
                if day['MACD'] > previous['MACD'] and MACD_signals[-1] == 1:
                    MACD_signals.append(1)
                else:
                    MACD_signals.append(0)
    
    df['MACD_signals'] = MACD_signals

    previous_MACD_signal = []
    MACD_change = []
    signal_change = []
    
    for i in range(len(df)):
        if i == 0:
            previous_MACD_signal.append(0)
            MACD_change.append(0)
            signal_change.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            if day['MACD'] > previous['MACD']:
                MACD_change.append(1)
            else:
                MACD_change.append(0)
                
            if day['signal_line'] > previous['signal_line']:
                signal_change.append(1)
            else:
                signal_change.append(0)
                
            previous_MACD_signal.append(previous['MACD_signals'])
    
    convergence = []
    
    for i in range(len(df)):
        if i == 0:
            convergence.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            current_MACD_distance = day['MACD'] - day['signal_line']
            previous_MACD_distance = previous['MACD'] - previous['signal_line']
            
            convergence.append(current_MACD_distance - previous_MACD_distance)
        
    df['previous_MACD_signal'] = previous_MACD_signal        
    df['MACD_change'] = MACD_change  
    df['signal_change'] = signal_change  
    df['convergence'] = convergence
    
    print('MACD Variables Calculated')
    print('')
    df = df[::-1]      
    return df

def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    u.ewm(com=period-1,adjust=False).mean()
    rs = u.ewm(com=period-1,adjust=False).mean() / d.ewm(com=period-1,adjust=False).mean()
    return 100 - 100 / (1 + rs)

def add_RSI(df, nonadj_Close):
    
    df = df[::-1]
    
    print('Calculating RSI')
    
    df['RSI'] = RSI(df['Close'], 14)    
    
    print('RSI Calculated')
    print('')
    
    stochs = []
    
    lows = []
    highs = []
    
    for i in range(len(df)):
        
        ratio = df.iloc[i,:]['Close'] / nonadj_Close[i]
        lows.append(ratio*df.iloc[i,:]['Low'])
        highs.append(ratio * df.iloc[i,:]['High'])
            
    df['High'] = highs
    df['Low'] = lows
    
    print(df)
    
    print('Calculating Stochastic RSI')
    for i in range(len(df)):
        
        if i <= 12:
            stochs.append(0.0)
        else: 
            close  = df.iloc[i,:]['Close']
            
            last14 = df.iloc[(i-13):(i+1),:]
            
            lowest = min(list(last14['Low']))
            highest = max(list(last14['High']))
            
            stoch = 100*(close - lowest) / (highest - lowest)
                
            stochs.append(stoch)
    
    df['Stochastic_Oscillator'] = stochs
    
    df['stoch_ma'] = df['Stochastic_Oscillator'].ewm(span=3,adjust=False).mean()
    
    
    
    
    STOCH_signals = []
    for i in range(len(df)):
        
        if i <= 1:
            STOCH_signals.append(0)
        
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            previous_two = df.iloc[i-2,:]
            
        
            if day['Stochastic_Oscillator'] > previous['Stochastic_Oscillator'] and previous['Stochastic_Oscillator'] < previous_two['Stochastic_Oscillator']:
                STOCH_signals.append(1)
            else:
                if day['Stochastic_Oscillator'] > previous['Stochastic_Oscillator'] and STOCH_signals[-1] == 1:
                    STOCH_signals.append(1)
                else:
                    STOCH_signals.append(0)
    
    df['STOCH_signals'] = STOCH_signals

    previous_STOCH_signal = []
    STOCH_change = []
    STOCH_ma_change = []
    
    for i in range(len(df)):
        if i == 0:
            previous_STOCH_signal.append(0)
            STOCH_change.append(0)
            STOCH_ma_change.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            if day['Stochastic_Oscillator'] > previous['Stochastic_Oscillator']:
                STOCH_change.append(1)
            else:
                STOCH_change.append(0)
                
            if day['stoch_ma'] > previous['stoch_ma']:
                STOCH_ma_change.append(1)
            else:
                STOCH_ma_change.append(0)
                
            previous_STOCH_signal.append(previous['Stochastic_Oscillator'])
    
    stoch_convergence = []
    
    for i in range(len(df)):
        if i == 0:
            stoch_convergence.append(0)
        else:
            day = df.iloc[i,:]
            previous = df.iloc[i-1,:]
            
            current_STOCH_distance = day['Stochastic_Oscillator'] - day['stoch_ma']
            previous_STOCH_distance = previous['Stochastic_Oscillator'] - previous['stoch_ma']
            
            stoch_convergence.append(current_STOCH_distance - previous_STOCH_distance)
        
    df['previous_STOCH_signal'] = previous_STOCH_signal        
    df['STOCH_change'] = STOCH_change  
    df['STOCH_ma_change'] = STOCH_ma_change  
    df['stoch_convergence'] = stoch_convergence
    df.to_csv('Stochastics.csv')
    print('Stochastic Variables Calculated')
    
    df = df[::-1]
    
    return df
    
    
    