# library import
import pandas as pd
import numpy as np
import scipy.stats

def drawdown(return_series: pd.Series):
    """
    Take a time series of asset returns.
    returns a DataFrame with columns for the wealth index,
    the previous peaks and the percentae drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod() # cumultavie product
    previous_peaks = wealth_index.cummax() # cummulative max
    drawdowns  = (wealth_index - previous_peaks) / previous_peaks # get the negative return, loss, subject to the previous peak
    return pd.DataFrame(
        {
            "Wealth": wealth_index,
            "Peaks": previous_peaks,
            "Drawdown": drawdowns
        }
    )
    

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, index_col=0, na_values=-99.99)
    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = [
        "SmallCap",
        "LargeCap"
    ]
    rets = rets / 100
    rets.index = pd.to_datetime(
        rets.index,
        format="%Y%m" # format of the given datetime, in this case, year and month, e.g. 192507 -> July of 1925
    ).to_period('M') # get the monthly return in the form of ratio.
    return rets


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                     header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Compute the skewness of the given Series or Dataframe based on the given equation above.
    '''
    # (R - E(R))
    demeaned_r = r - r.mean()

    '''
    # use the population standard deviation, so set ddof=0
    # where ddof stands for "delta degree of freedom"
    # The divisor used in calculations is (N - ddof), where N represents the number of elements.
    '''
    # s.d.(R)
    sigma_r = r.std(ddof=0)

    # ExpectedValue(demeaned_r ** 3)
    exp = (demeaned_r**3).mean()
    return exp/((sigma_r)**3)


def kurtosis(r):
    '''
    alternative of the scipy.stats.kutosis
    '''
    # (R - E(R))
    demeaned_r = r - r.mean()

    # Var(R)
    sigma_r = r.std(ddof=0)

    # ExpectedValue(demeaned_r ** 4)
    exp = (demeaned_r**4).mean()
    return  exp/(sigma_r**4)


def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if the Series is normally distributed or not.
    Test is applied at the 1% level by default and the threshhold can be altered as due course.
    Return True if the hypothesis of normality is accepted, False otherwise.
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level