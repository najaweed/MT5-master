import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import numpy as np
import MetaTrader5 as mt5
from statsmodels.tsa.vector_ar.var_model import VAR, FEVD

import statsmodels.tsa as tsa
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, ccf, ccovf, kpss, coint
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def low_pass_filter(xs, alpha=0.8):
    # low_cut = band[0] / (len(xs) / 2)

    xs = np.concatenate((xs, xs[::-1]))
    sig = signal.butter(N=4, Wn=alpha, btype='lowpass')
    b = sig[0]
    a = sig[1]
    y1 = signal.filtfilt(b, a, xs)

    xs = y1[:int(len(xs) / 2)]
    return xs


def forex_smoother(x_df: dict):
    a_range = np.linspace(0.001, 0.999, 1000)

    s_x = 0
    for cur, df in x_df.items():
        s_x += df.close.to_numpy()

    print('raw = ', np.mean(s_x))
    raw_mean = np.mean(s_x)
    alpha_x = 0.1
    smooth = {}

    for a in a_range:
        s_x = 0
        for cur, df in x_df.items():
            smooth[f'{cur}'] = low_pass_filter(df.close.to_numpy(), alpha=a)
            s_x += smooth[f'{cur}']

        if abs(np.mean(s_x) / 10) < abs(raw_mean):
            print(np.mean(s_x))
            print('alpha = ', a)
            alpha_x = a
            break
    return smooth


def xforex_smoother(x_df: dict,alpha = 0.1):

    smooth = pd.DataFrame(columns=x_df.keys())

    for sym,df in x_df.items():

        smooth[f'{sym}'] = pd.DataFrame(low_pass_filter(df.close.to_numpy(), alpha=alpha),index=df.index)

    return smooth



class ForcastVAR:
    def __init__(self,
                 x_df: pd.DataFrame):
        self.train_raw, self.test_raw = self.split_df(x_df)
        self.train, self.test = self.train_raw.diff().dropna(), self.test_raw.diff().dropna()

    @staticmethod
    def split_df(df, split_ratio=0.02):
        index_split = int(len(df) * (1 - split_ratio))
        return df.iloc[:index_split, ], df.iloc[index_split - 1:, ]

    def forcast(self, step=30, lag=None):
        x_model = VAR(self.train.to_numpy(),)
        lag_len = int(len(self.train) * 0.8) if lag is None else lag

        var_model = x_model.fit(lag_len)
        forcast = var_model.forecast(self.train.values, steps=len(self.test) + step)
        index_x = pd.date_range(self.test.index[0],periods=len(self.test) + step,freq='1min')

        forecast_var = pd.DataFrame(forcast,
                                    columns=self.train.columns,
                                    index=index_x)
        return forecast_var

    def recon_price(self, ):
        diff_predict_price = self.forcast()
        predict_price = pd.DataFrame(columns=diff_predict_price.columns,index=diff_predict_price.index)

        for sym in self.test_raw.columns:
            for i, inx in enumerate(predict_price.index):
                if i == 0:
                    predict_price[sym][i] = self.test_raw[sym][0]
                else:
                    predict_price[sym][i] = predict_price[sym][i-1] + diff_predict_price[sym][i]
        return predict_price
from LiveRate import ForexMarket

currenciesx = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']  # , 'CAD', 'AUD', ]  # 'NZD']
fx = ForexMarket(currenciesx)#.gen_fx_indexes(time_shift=400, time_frame=mt5.TIMEFRAME_M1)
xt = fx.get_all_rates(time_shift=400, time_frame=mt5.TIMEFRAME_M1)
sx = xforex_smoother(xt).fillna(method='bfill')
ff = ForcastVAR(sx)

# def _recon_price(raw_price:pd.DataFrame,diff_predict_price:pd.DataFrame):
#     predict_price = pd.DataFrame(columns=diff_predict_price.columns,index=diff_predict_price.index)
#
#     for sym in raw_price.columns:
#         for i, inx in enumerate(predict_price.index):
#             if i == 0:
#                 predict_price[sym][i] = raw_price[sym][0]
#             else:
#                 predict_price[sym][i] = predict_price[sym][i-1] + diff_predict_price[sym][i]
#     return predict_price
# print(_recon_price(ff.test_raw,forcasti))
rec_pirce = ff.recon_price()#_recon_price(ff.test_raw,forcasti)
for fc in rec_pirce.columns:
    plt.figure(fc)
    plt.plot(ff.train_raw[fc])
    plt.plot(ff.test_raw[fc])

    plt.plot(xt[fc].close)
    plt.plot(rec_pirce[fc])

    # plt.figure(fc+'ii')
    # plt.plot(forcasti[fc])
    # plt.plot(forcasti[fc])
plt.show()


