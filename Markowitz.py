import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RF_RATE = 0.03


def opt_weight(r, cov_mat, mu_vec):
    """
   Given a desired rate of expected return, r, and a nxn covariance matrix, COV_MAT, and nx1 expected
   returns vector, MU_VEC, finds the Efficient Frontier portfolio with the desired return. Returns
   a nx1 weight vector, w, with weights corresponding to the assets in the same order as MU_VEC.
    """
    ones = np.ones(len(mu_vec)).reshape((len(mu_vec), 1))
    A = mu_vec.T.dot(np.linalg.inv(cov_mat)).dot(mu_vec)[0][0]
    B = ones.T.dot(np.linalg.inv(cov_mat)).dot(ones)[0][0]
    C = mu_vec.T.dot(np.linalg.inv(cov_mat)).dot(ones)[0][0]
    
    
    nu = (2*A - 2*r*C) / (A*B - C*C)
    lam = (2*r - nu * C) / A
    w = (1/2) * np.linalg.inv(cov_mat).dot(lam*mu_vec + nu*ones)
    return w

def get_vol(w, cov_mat, annualized=False):
    """
   Given a covariance matrix, COV_MAT, returns the volatility of the portfolio defined by the
   nx1 weight matrix W. If COV_MAT contains covariances based on daily returns, ANNUALIZED may be
   used to annualize the output.
    """
    variance = w.T.dot(cov_mat).dot(w)[0][0]
    if annualized:
        variance = variance*252
    
    volatility = np.sqrt(variance)
    return volatility

def get_MVP(mu_vec, cov_mat, mark_vols, mark_rets, stocks):
    """
    Given the efficient frontier volatilities and returns, MARK_VOLS and MARK_RETS, corresponding
    to the x- and y-axis of the Efficient Frontier for a portfolio with expected returns vector
    MU_VEC and covariance matrix COV_MAT, will return the Minimum Variance Portfolio. For labeling,
    STOCKS should be a list of asset names corresponding to the assets in MU_VEC (and in the same
    order).
    """
    MVP_index = np.argmin(mark_vols)
    MVP = opt_weight(mark_rets[MVP_index], cov_mat, mu_vec)
    MVP_series = pd.Series(index=stocks, data=MVP.reshape((len(MVP))))
    print("Minimum Variance Portfolio:\n" + str(MVP_series.sort_values()))
    
    return MVP_series

def get_market_port(mu_vec, cov_mat, mark_vols, mark_rets, rf, stocks):
    """
    Given the efficient frontier volatilities and returns, MARK_VOLS and MARK_RETS, corresponding
    to the x- and y-axis of the Efficient Frontier for a portfolio with expected returns vector
    MU_VEC and covariance matrix COV_MAT, will return the closest portoflio to the market portfolio
    (aka Tangency Portfolio) given the risk-free rate RF. For labeling, STOCKS should be a list of
    asset names corresponding to the assets in MU_VEC (and in the same order).
    """
    derivatives = []
    for i in range(1, len(mark_rets) - 1):
        derivative = (mark_rets[i + 1] - mark_rets[i - 1]) / (mark_vols[i + 1] - mark_vols[i - 1])
        derivatives.append(derivative)
    
    cap_mkts_slopes = []
    for i in range(1, len(mark_rets) - 1):
        cap_mkts_slope = (mark_rets[i] - rf) / mark_vols[i]
        cap_mkts_slopes.append(cap_mkts_slope)
    
    MVP_index = np.argmax(derivatives)
    market_portfolio_index = np.argmin((np.array(derivatives[MVP_index:]) - np.array(cap_mkts_slopes[MVP_index:]))**2)
    market_portfolio_index += MVP_index
    market_portfolio_ret = mark_rets[market_portfolio_index - 1]
    market_portfolio = opt_weight(market_portfolio_ret, cov_mat, mu_vec)
    market_port_series = pd.Series(index=stocks, data=market_portfolio.reshape((len(market_portfolio))))
    
    cap_mkt_slope = cap_mkts_slopes[market_portfolio_index]
    
    print("Market Portfolio, assuming risk-free rate of %.2f:\n" % rf + str(market_port_series.sort_values()))
    
    return market_port_series, cap_mkt_slope

def plot_eff_front(mu_vec, cov_mat, stocks):
    """
    Given the expected returns vector and covariance matrix, MU_VEC and COV_MAT, for the assets whose
    in STOCKS (a list of asset names), will plot the assets on a return-volatility graph, draw the
    associated Markowitz Efficient Frontier, print the MVP, find and print the Market Portfolio, and
    plot the capital markets line.
    """
    mu_vec = mu_vec.reshape((len(mu_vec), 1))
    rets = np.linspace(-2 * np.abs(min(mu_vec)), 2*np.abs(max(mu_vec)), 100)
    vols = [get_vol(opt_weight(r, cov_mat, mu_vec), cov_mat) for r in rets]
    
    MVP = get_MVP(mu_vec, cov_mat, vols, rets, stocks)
    market_port, cap_mkt_slope = get_market_port(mu_vec, cov_mat, vols, rets, RF_RATE, stocks)
    x = np.linspace(0, max(vols), 100)
    cap_mkts_line = [RF_RATE + cap_mkt_slope*x for x in x]
    
    plt.plot(vols, rets)
    plt.plot(x, cap_mkts_line)
    plt.xlim(xmin=0)
    
    return MVP, market_port
