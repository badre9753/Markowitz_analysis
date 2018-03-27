from Markowitz import *
from stock_scraper import *

import numpy as np

prices_panel = get_stock_prices(['FRAN', 'F', 'VC', 'PSP', 'SPY', 'QQQ'], '2015-01-01', '2018-03-23')
prices = prices_panel['Adj Close']

rets = prices.pct_change().dropna()
cov_mat = rets.cov()
mu_vec = rets.mean()

plot_eff_front(np.array(mu_vec).reshape((len(mu_vec), 1)), np.array(cov_mat), mu_vec.index)
