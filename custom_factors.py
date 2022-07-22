"""
Custom factors to be fed to the pipeline in the LSTM algorithm and functions
to retreive data than cannot be retreived from the pipeline
"""

# Is it possible to modify the factors so they take as inputs "from and to"
# rather than creating classes for each one?

import numpy as np
import pandas as pd
from pipeline_live.data.alpaca.pricing import USEquityPricing
from pipeline_live.data.alpaca.factors import Returns
from pipeline_live.data.iex.factors import CustomFactor
from datetime import datetime
from datetime import timedelta
import os

class ComputeSTR_t1(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 5
    def compute(self, today, assets, out, close):
        out[:] = (close[-1] / close[-4])

class ComputeSTR_t2(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 6
    def compute(self, today, assets, out, close):
        out[:] = (close[-2] / close[-5])

class ComputeSTR_t3(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 7
    def compute(self, today, assets, out, close):
        out[:] = (close[-3] / close[-6])

class ComputeSTR_t4(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 8
    def compute(self, today, assets, out, close):
        out[:] = (close[-4] / close[-7])

class ComputeSTR_t5(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 9
    def compute(self, today, assets, out, close):
        out[:] = (close[-5] / close[-8])

class ComputeSTR_t6(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 10
    def compute(self, today, assets, out, close):
        out[:] = (close[-6] / close[-9])

class ComputeSTR_t7(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 11
    def compute(self, today, assets, out, close):
        out[:] = (close[-7] / close[-10])

class ComputeSTR_t8(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 12
    def compute(self, today, assets, out, close):
        out[:] = (close[-8] / close[-11])

class ComputeSTR_t9(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 13
    def compute(self, today, assets, out, close):
        out[:] = (close[-9] / close[-12])

class ComputeSTR_t10(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 14
    def compute(self, today, assets, out, close):
        out[:] = (close[-10] / close[-13])


########################################################################33


class ComputeMOM12_t1(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 249
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-23] / close[-248])

class ComputeMOM12_t2(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 250
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-24] / close[-249])

class ComputeMOM12_t3(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 251
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-25] / close[-250])

class ComputeMOM12_t4(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-26] / close[-251])

class ComputeMOM12_t5(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 253
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-27] / close[-252])

class ComputeMOM12_t6(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 254
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-28] / close[-253])

class ComputeMOM12_t7(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 255
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-29] / close[-254])

class ComputeMOM12_t8(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 256
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-30] / close[-255])

class ComputeMOM12_t9(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 257
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-31] / close[-256])

class ComputeMOM12_t10(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 258
    def compute(self, today, assets, out, close):
        out[:] = np.log(close[-32] / close[-257])


########################################################


class ComputeBAB_t1(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 26
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-24:], axis=0)+1))

class ComputeBAB_t2(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 27
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-25:-1], axis=0)+1))

class ComputeBAB_t3(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 28
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-26:-2], axis=0)+1))

class ComputeBAB_t4(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 29
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-27:-3], axis=0)+1))

class ComputeBAB_t5(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 30
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-28:-4], axis=0)+1))

class ComputeBAB_t6(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 31
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-29:-5], axis=0)+1))

class ComputeBAB_t7(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 32
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-30:-6], axis=0)+1))

class ComputeBAB_t8(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 33
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-31:-7], axis=0)+1))

class ComputeBAB_t9(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 34
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-32:-8], axis=0)+1))

class ComputeBAB_t10(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 35
    def compute(self, today, assets, out, close):
        out[:] = np.log(np.log(np.nanstd(close[-33:-9], axis=0)+1))

####################################################################

def compute_iret(df, str_col, ind_col='ind'):
    data_stock = df[[str_col, ind_col]]
    data_stock.columns = ['STR', 'ind']
    data_industry = pd.DataFrame(index=data_stock.index,
                                 data={"ret_ind": data_stock.groupby("ind").\
                                       transform(np.mean).values.flatten()})
    iret = data_industry["ret_ind"]
    return iret

def compute_istr(df, str_col, ind_col='ind'):
    data_stock = df[[str_col, ind_col]]
    data_stock.columns = ['STR', 'ind']
    data_industry = pd.DataFrame(index=data_stock.index,
                                 data={"STR_ind": data_stock.groupby("ind").\
                                       transform(np.mean).values.flatten()})
    istr = data_stock["STR"]/data_industry["STR_ind"]
    return istr
