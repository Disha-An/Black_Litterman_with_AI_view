import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime as dt
from stockstats import StockDataFrame as Sdf

# indicator generae function ###

def get_inputs(etf):
    ETF_df = Sdf.retype(etf)
    
	# MACD
    macd = pd.DataFrame(ETF_df.get('macd'))
    macd = macd[19:]	
	# CR indicator, including 5, 10, 20 days moving average
    cr = pd.DataFrame(ETF_df['cr'])
    cr = cr[19:] 
	# cr ma 1
    cr_ma1 = pd.DataFrame(ETF_df['cr-ma1'])
    cr_ma1 = cr_ma1[19:]
    # cr ma 2
    cr_ma2 = pd.DataFrame(ETF_df['cr-ma2'])
    cr_ma2 = cr_ma2[19:]
    # cr ma 3
    cr_ma3 = pd.DataFrame(ETF_df['cr-ma3'])
    cr_ma3 = cr_ma3[19:] 
    # kdjd
    kdjd = pd.DataFrame(ETF_df['kdjd'])
    kdjd = kdjd[19:]   
    # kdjj
    kdjj = pd.DataFrame(ETF_df['kdjj'])
    kdjj = kdjj[19:]
    # MACD signal line
    macds = pd.DataFrame(ETF_df['macds'])
    macds = macds[19:]  
    # MACD histogram
    macdh = pd.DataFrame(ETF_df['macdh'])
    macdh = macdh[19:] 
    # bolling
    bolling = pd.DataFrame(ETF_df['boll'])
    bolling = bolling[19:]
    # 6 days RSI
    rsi6 = pd.DataFrame(ETF_df['rsi_6'])
    rsi6 = rsi6[19:]
    # 12 days RSI
    rsi12 = pd.DataFrame(ETF_df['rsi_12'])
    rsi12 = rsi12[19:]
    # 10 days WR
    wr10 = pd.DataFrame(ETF_df['wr_10'])
    wr10 = wr10[19:]
    # 6 days WR
    wr6 = pd.DataFrame(ETF_df['wr_6'])
    wr6 = wr6[19:]
    # 14 days cci
    cci14 = pd.DataFrame(ETF_df['cci'])
    cci14 = cci14[19:]
    # 20 days cci
    cci20 = pd.DataFrame(ETF_df['cci_20'])
    cci20 = cci20[19:]   
    # TR (true range)
    tr = pd.DataFrame(ETF_df['tr'])
    tr = tr[19:]    
    # ATR (Average True Range)
    atr = pd.DataFrame(ETF_df['atr'])
    atr = atr[19:]
    # DMA difference of 10 and 50 moving average
    dma = pd.DataFrame(ETF_df['dma'])
    dma = dma[19:]    
    # +DI default to 14 days
    pdi = pd.DataFrame(ETF_df['pdi'])
    pdi = pdi[19:]   
    # -DI default to 14 days
    mdi = pd.DataFrame(ETF_df['mdi'])
    mdi = mdi[19:]   
    # DX, default to 14 days of +DI and -DI
    dx = pd.DataFrame(ETF_df['dx'])
    dx = dx[19:]   
    # TRIX default to 12 days
    trix = pd.DataFrame(ETF_df['trix'])
    trix = trix[19:]
    # VR, default to 26 days
    vr = pd.DataFrame(ETF_df['vr'])
    vr = vr[19:]
	
    logreturn = pd.DataFrame(np.log(ETF_df['adj close']).diff())
    l1 = logreturn[0:len(logreturn)-19]
    l2 = logreturn[1:len(logreturn)-18]
    l3 = logreturn[2:len(logreturn)-17]
    l4 = logreturn[3:len(logreturn)-16]
    l5 = logreturn[4:len(logreturn)-15]
    l6 = logreturn[5:len(logreturn)-14]
    l7 = logreturn[6:len(logreturn)-13]
    l8 = logreturn[7:len(logreturn)-12]
    l9 = logreturn[8:len(logreturn)-11]
    l10 = logreturn[9:len(logreturn)-10]
    l11 = logreturn[10:len(logreturn)-9]
    l12 = logreturn[11:len(logreturn)-8]
    l13 = logreturn[12:len(logreturn)-7]
    l14 = logreturn[13:len(logreturn)-6]
    l15 = logreturn[14:len(logreturn)-5]
    l16 = logreturn[15:len(logreturn)-4]
    l17 = logreturn[16:len(logreturn)-3]
    l18 = logreturn[17:len(logreturn)-2]
    l19 = logreturn[18:len(logreturn)-1]
    l20 = logreturn[19:len(logreturn)-0]
    
    fin_df = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,macd,cr,cr_ma1,cr_ma2,cr_ma3,kdjd,kdjj,macds,macdh,bolling,rsi6,rsi12,wr10,wr6,cci14,cci20,tr,atr,dma,pdi,mdi,dx, trix, vr),axis=1)
    fin_df = pd.DataFrame(fin_df,columns = ["l1","l2","l3","l4","l5","l6","l7","l8","l9","l10","l11","l12","l13","l14","l15","l16","l17","l18","l19","l20","macd","cr","cr_ma1","cr_ma2","cr_ma3","kdjd","kdjj","macds","macdh","bolling","rsi6","rsi12","wr10","wr6","cci14","cci20","tr","atr","dma","pdi","mdi","dx", "trix","vr"])
    return fin_df

	
start = dt.datetime(2010,1,1)
end = dt.datetime(2017,10,31)

# ETFs list : IJK, IVV, IWP, IWV, IYY, LQD, VOT
## LQD
sticker = 'LQD'
etf_data = pdr.get_data_yahoo(sticker, start, end)
# get the date data
date = []
for i in range(len(etf_data.index)):
    ts = pd.to_datetime(str(etf_data.index.values[i]))
    d = ts.strftime("%Y-%m-%d")
    date.append(d)
# get the indicator data
all_data = get_inputs(etf_data)
# reformat the data set and remove the nas
date = date[19:]
all_data["Date"]=date
result = all_data.dropna()
result.to_csv('LQD.csv')