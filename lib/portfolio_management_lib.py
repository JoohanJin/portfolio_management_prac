# library import
from typing import Union, Literal
import pandas as pd
import numpy as np
import scipy.stats


# assume that the data we are handling is stock data for now, not cryptocurrencies or other assets.
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
    expected_return = mean_return**(interval) - 1
    return expected_return


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
    

def get_data(
    csv_name: str
):
    """
    Load the Dataset based on the csv file name
    """
    data = pd.read_csv(
        csv_name,
        header = 0,
        index_col = 0,
        na_values = -99.99,
    )
    rets = data/100
    rets.index = pd.to_datetime(
        rets.index,
        format = "%Y%m"
    ).to_period("M")

    return rets


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
        r: pd.DataFrame | pd.Series | float,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    """
    Annualizes a set of returns
    We should infer the period per year
    """
    if isinstance(r, pd.DataFrame) or isinstance(r, pd.Series):
        compounded_growth = (1 + r).prod() # prod(): return the product of the values over the requested axis.
        n_periods = r.shape[0]
        periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
        return compounded_growth ** (periods_per_year/n_periods) - 1
    elif isinstance(r, float):
        growth = 1 + r
        periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
        return growth ** (periods_per_year) - 1
    else:
        raise TypeError

def annualize_vol(
        r: pd.DataFrame | pd.Series | float,
        freq: Union[Literal['Daily', 'Monthly', 'Quarterly', 'Annual']]= 'Monthly',
    ):
    """
    Annualizes the volatility of a set of returns
    We should infer the period per year
    """
    if isinstance(r, pd.DataFrame) or isinstance(r, pd.Series):
        periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
        return r.std() * (periods_per_year ** 0.5)
    elif isinstance(r, float) or isinstance(r, int):
        periods_per_year: int = 255 if freq == 'Daily' else 12 if freq == 'Monthly' else 4 if freq == 'Quarterly' else 1 # last one is Annual
        return r * (periods_per_year ** 0.5)
    else:
        raise TypeError


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


def portfolio_ret(weights, returns):
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


def plot_ef2(
        expected_return,
        cov,
        n_points = 20
    ):
    """
    Plots the 2-asset efficient frontier.
    """
    if expected_return.shape[0] != 2:
        raise ValueError
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_ret(w, expected_return) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({"R": rets, "V": vols})
    return ef.plot.line(x = "V", y = "R", style = ".-")


from scipy.optimize import minimize

def minimize_vol(
    target_return,
    expected_return, 
    cov
    ):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix.
    """
    n: int = expected_return.shape[0] # get the number of assets for the portfolio
    init_guess = np.repeat(1/n, n) # each asset has the same weights in the portfolio construction
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples

    # construct the constraints
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    return_is_target = {
        'type': 'eq',
        'args': (expected_return,),
        'fun': lambda weights, expected_return: target_return - portfolio_ret(weights, expected_return)
    }

    # Now we minimze the volatility for the given level of returns.
    weights = minimize(
        portfolio_vol,
        init_guess,
        args = (cov,),
        method = 'SLSQP', # Minimize a scalar function of one or more varaibles using Sequential Least Squares Programming (SLSQP).
        options = {'disp': False},
        constraints = (weights_sum_to_1, return_is_target),
        bounds = bounds
    )

    return weights.x


def optimal_weights(
    expected_return,
    cov,
    n_points = 20,
):
    """
    Get the weight of each assets for the best possible cases, either the lowest volatility or the highest return
    from the given data.
    """
    target_returns = np.linspace(expected_return.min(), expected_return.max(), n_points)
    weights = [minimize_vol(target_return, expected_return, cov) for target_return in target_returns]
    return weights


def plot_ef(
    expected_return,
    cov,
    n_points = 20,
    show_cml = True,
    show_ew = True,
    show_gmv = True,
    riskfree_rate = 0.1,
):
    """
    Plots the ulti-asset efficient frontier
    """
    weights = optimal_weights(expected_return, cov, n_points = n_points) # not yet implemented
    rets = [portfolio_ret(w, expected_return) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "R": rets,
        "V": vols,
    })

    ax = ef.plot.line(x = "V", y = "R", style = "-")

    if show_cml:
        ax.set_xlim(left = 0)

        # get MSR
        weight_msr = msr(
            expected_return,
            cov,
            riskfree_rate = riskfree_rate
        )
        ret_msr = portfolio_ret(
            weights = weight_msr,
            returns = expected_return,
        )
        vol_msr = portfolio_vol(
            weights = weight_msr,
            covmat = cov,
        )

        # add CMl
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, ret_msr]
        ax.plot(
            cml_x, 
            cml_y, 
            color='green', 
            marker='o', 
            linestyle='dashed', 
            linewidth=2, 
            markersize=12
        )

    # show equally weighted portfolio to prevent the estimation error
    if show_ew:
        n = expected_return.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_ret(w_ew, expected_return)
        vol_ew = portfolio_vol(w_ew, cov)

        # add EW
        ax.plot(
            [vol_ew], 
            [r_ew], 
            color = "goldenrod", 
            marker = 'o', 
            markersize = 10
        )

    if show_gmv:
        weight_gmv = gmv(cov)
        ret_gmv = portfolio_ret(weight_gmv, expected_return)
        vol_gmv = portfolio_vol(weight_gmv, cov)

        # add GMV
        ax.plot(
            [vol_gmv],
            [ret_gmv],
            color = "midnightblue",
            marker = 'o',
            markersize = 10,
        )

    return ax


def msr(
    expected_return,
    cov,
    riskfree_rate = 0.01,
):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariane matrix.
    """
    n = expected_return.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n # n-tuple of 2d array
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe(
        weights,
        riskfree_rate,
        expected_return,
        cov
    ):
        """
        Returns the negative of the Sharpe ratio
        of the given portfolio
        """
        r = portfolio_ret(weights, expected_return)
        vol = portfolio_vol(weights, cov)
        return - (r - riskfree_rate) / vol # return sharpe raio

    weights = minimize(
        neg_sharpe,
        init_guess,
        args = (riskfree_rate, expected_return, cov),
        method = "SLSQP",
        options = {"disp": False},
        constraints = (weights_sum_to_1),
        bounds = bounds,
    )
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix.
    """
    n = cov.shape[0]
    return msr(np.repeat(1, n), cov, 0)


def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios average number of Firms.
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolio Average Size (market capitalization)
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind


def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index.
    """
    # get number of firms for each industry in the timeseries
    ind_nfirms = get_ind_nfirms()
    # get the market size of each industry
    ind_size = get_ind_size()
    # get the monthly return in each industry stock
    ind_return = get_ind_returns()
    # get the market capitalization of each industry
    ind_mktcap = ind_nfirms * ind_size
    # get the total market capitalization in timeseries
    total_mktcap = ind_mktcap.sum(axis = 1)
    # get the weight of each industry in terms of the toal market cap
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 0)
    total_market_return = (ind_capweight * ind_return).sum(axis = 1)
    return total_market_return


def compound(r) -> float | pd.Series:
    """
    returns the result of compunding the set of returns in r
    """
    return np. expm1(np.log1p(r).sum())


def run_cppi(
    risky_r,
    safe_r = None,
    m = 3, # multiplier
    start = 1000, # initial value
    floor = 0.8, # floor value for the value that I am holding
    riskfree_rate = 0.03,
    drawdown = 0,
) -> dict:
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    datetime = risky_r.index
    n_steps: int = len(datetime)
    account_value = start
    floor_value = start * floor
    peak = account_value


    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12

    
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if (drawdown):
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # recompute the new account value at the end of this step.
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak

    risky_wealth = start * (1 + risky_r).cumprod()

    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r,
    }
    return backtest_result


def summary_stats(r, riskfree_rate = 0.03) -> pd.DataFrame:
    """
    A convenience function to provide summary statistics on a set of returns
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r.
    """
    ann_r = r.aggregate(
        annualize_rets, 
        freq = "Monthly"
    )
    ann_vol = r.aggregate(
        annualize_vol, 
        freq = "Monthly"
        )
    ann_sr = r.aggregate(
        sharpe_ratio, 
        riskfree_rate = riskfree_rate
    )
    dd = r.aggregate(
        lambda r: drawdown(r).Drawdown.min()
        )
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(
        var_gaussian, modified = True
        )
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Volatility": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish_Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
    })