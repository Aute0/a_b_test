'''
This file contain 3 function
'''

from typing import Tuple
from scipy import stats
import numpy as np


def cpc_sample(n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float) -> np.ndarray:
    '''

    Parameters
    ----------
    n_samples - for norm and binom distribution
    cvr - for binom distribution
    reward_avg - for norm dist (as mean or loc)
    reward_std - for norm dist (as std or scale)

    Returns
    -------

    '''
    cvr = stats.binom.rvs(1, conversion_rate, size=n_samples)
    cpa = stats.norm.rvs(loc=reward_avg, scale=reward_std, size=n_samples)

    return cvr * cpa


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05) -> Tuple[bool, float]:
    result = stats.ttest_ind(cpc_a, cpc_b)
    if result[1] < alpha:
        return True, result[1]
    else:
        return False, result[1]


def aa_test(
        n_simulations: int,
        n_samples: int,
        conversion_rate: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    type_1_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate two cpc samples with the same cvr, reward_avg, and reward_std
        cpc_a = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        cpc_b = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)

        # Check t-test and save type 1 error

        type_1_errors[i] = t_test(cpc_a, cpc_b, alpha)[0]

        # Calculate the type 1 errors rate

        type_1_errors_rate = np.mean(type_1_errors)

    return type_1_errors_rate
