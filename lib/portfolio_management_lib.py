# library import
from typing import Union, Literal
import pandas as pd
import numpy as np
import scipy.stats


# assume that the data we are handling is stock data, not cryptocurrencies or other assets.
# I can diversify the type of data in the future.

def calculate_return_annualized(
        return_series: pd.DataFrame,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    """
    Compute the annualized return from a given time series of returns.
    """
    # TODO: implement the functino where get the number of working days in "the" year specifically to automatically get the
    # value of the interval, if freq is 'daily'
    interval: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual

    # get the number of periods
    n: int = return_series.shape[0]

    # get the mean return
    mean_return = (1 + return_series).prod()**(1/n)

    # get the annualized return
    annualized_return = mean_return**(interval) - 1
    return annualized_return


def calculate_volatility_annualized(
        return_series: pd.Series,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    interval: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
    annaulized_volatility = return_series.std() * np.sqrt(interval)
    return annaulized_volatility



def drawdown(return_series: pd.Series):
    """
    Take a time series of asset returns.
    returns a DataFrame with columns for the wealth index,
    the previous peaks and the percentae drawdown
    """
    if isinstance(return_series, pd.Series):
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
    elif isinstance(return_series, pd.DataFrame):
        return_series.aggregate(drawdown, index = 0)
    else:
        raise TypeError
    

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv(
        "data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header = 0,
        index_col = 0,
        na_values = -99.99,
    )
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
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv",
        header = 0,
        index_col = 0,
        parse_dates = True
    )
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


def semideviation(
    r
):
    """
    Returns the semi-deviation aka negative semi-deviatino of r
    r must be a Series of a DataFrame, else raises a TypeError.
    """
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)


def var_historic(r, level = 5):
    """
    Returns the history Value at Risk at a specified level.
    i.e. returns the number such that 'level' percent of the returns fall below that number, and the (100 - level ) percent are above.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, axis = 0, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) 
    else:
        raise TypeError("Expected r to be a Series or DataFrame.")

    
def cvar_historic(r, level = 5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, axis = 0, level = level)
    elif isinstance(r, pd.Series):
        return -r[r <= - var_historic(r, level = level)].mean()
    else:
        raise TypeError("r should be either the Series or the DataFrame.")


from scipy.stats import norm

def var_gaussian(
        r,
        level = 5,
        modified = False, # modified parameter to decide to modify the z value to cornish_fisher or not.
    ):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame.
    if "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification.
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if (modified):
        s = skewness(r)
        k = kurtosis(r)
        # calculate the cornish_fisher adjusted z-score based on the equation: new z score based on the Cornish-Fisher analysis. 
        z = (z +
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z * r.std(ddof=0))


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def annualize_rets(
        r: pd.DataFrame | pd.Series,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',

    ):
    """
    Annualizes a set of returns
    We should infer the period per year
    """
    compounded_growth = (1 + r).prod() # prod(): return the product of the values over the requested axis.
    n_periods = r.shape[0]
    periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
    return compounded_growth ** (periods_per_year/n_periods) - 1


def annualize_vol(
        r: pd.DataFrame | pd.Series,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    """
    Annualizes the volatility of a set of returns
    We should infer the period per year
    """
    periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio(
        r: pd.DataFrame | pd.Series,
        riskfree_rate: float = 0.05,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    """
    Computes the annualized sharpe ratio of a set of returns.

    A measure used in finanace to evaluate the performance of an investment by comparing its retrun to its risk.

    sharpe_ratio = ((return of the portfolio) - (risk-free rate))/(standard deviation of the portfolio's excess return)
    """
    # convert the annual riskfree rate to per period
    periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
    rf_per_period = (1 + riskfree_rate) ** (1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def portfolio_returns(weights, returns):
    """
    Compute the return on a portfolio from consituent returns and weights.
    Weights are a numpy array or N x 1 Matrix and returns are numpy array or Nx1 Matrix.
    """
    return weights.T @ returns
    

def portfolio_vol(weights, covmat):
    """
    Will return s.d. not a volatility
    Computes the vol of a portfolio from a covariance matrix and constitutent weight.
    Weights are a numpy array or N x 1 matrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights) ** 0.5