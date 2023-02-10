'''
This file contain 2 function
'''

from typing import Tuple
from scipy import stats
import numpy as np

def cpc_sample(
    n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float) -> np.ndarray:
    '''

    Parameters
    ----------
    n_samples - for norm and binom distribution
    conversion_rate - for binom distribution
    reward_avg - for norm dist (as mean or loc)
    reward_std - for norm dist (as std or scale)

    Returns
    -------

    '''
    cvr = stats.binom.rvs(1, conversion_rate, size=n_samples)
    cpa = stats.norm.rvs(loc=reward_avg, scale=reward_std, size=n_samples)

    return cvr*cpa

def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05
) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    result = stats.ttest_ind(cpc_a, cpc_b)
    if result[1] < alpha:
        return True, result[1]
    else:
        return False, result[1]
