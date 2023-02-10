from typing import List, Tuple
import numpy as np
from scipy import stats

def cpc_sample(n_samples: int, cvr: float, reward_avg: float, reward_std: float) -> np.ndarray:
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
    cvr_dst = stats.binom.rvs(1, cvr, size=n_samples)
    cpa = stats.norm.rvs(loc=reward_avg, scale=reward_std, size=n_samples)

    return cvr_dst * cpa


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05) -> Tuple[bool, float]:
    result = stats.ttest_ind(cpc_a, cpc_b)
    if result[1] < alpha:
        return True, result[1]
    else:
        return False, result[1]


def aa_test(
        n_simulations: int,
        n_samples: int,
        cvr: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    type_1_errors = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Generate two cpc samples with the same cvr, reward_avg, and reward_std
        cpc_a = cpc_sample(n_samples, cvr, reward_avg, reward_std)
        cpc_b = cpc_sample(n_samples, cvr, reward_avg, reward_std)

        # Check t-test and save type 1 error

        type_1_errors[i] = t_test(cpc_a, cpc_b, alpha)[0]

        # Calculate the type 1 errors rate

        type_1_errors_rate = np.mean(type_1_errors)

    return type_1_errors_rate


def ab_test(
        n_simulations: int,
        n_samples: int,
        cvr: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/B test (simulation)."""

    type_2_errors = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Generate two cpc samples with the same cvr, reward_avg, and reward_std
        cpc_a = cpc_sample(n_samples, cvr, reward_avg, reward_std)
        cpc_b = cpc_sample(n_samples, cvr * (1 + mde), reward_avg, reward_std)

        # Check t-test and save type 2 error

        type_2_errors[i] = t_test(cpc_a, cpc_b, alpha)[0]

        # Calculate the type 1 errors rate

        type_2_errors_rate = 1 - np.mean(type_2_errors)

    return type_2_errors_rate

def select_sample_size(
        n_samples_grid: List[int],
        n_simulations: int,
        cvr: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[int, float, float]:
    """Select sample size."""

    if not n_samples_grid:
        raise RuntimeError("Can't find sample size")

    for n_samples in n_samples_grid:
        # Implement your solution here

        type_1_error = aa_test(n_simulations, n_samples, cvr, reward_avg, reward_std, alpha)

        type_2_error = ab_test(n_simulations, n_samples, cvr, mde, reward_avg, reward_std, alpha)

        if type_1_error <= alpha and type_2_error <= beta:
            return n_samples, type_1_error, type_2_error
        if n_samples == n_samples_grid[-1] and type_1_error > alpha and type_2_error > beta:
            raise RuntimeError("Make sure that the grid is big enough"
                                   f"last type 1 error: {type_1_error}, "
                                   f"last type 2 error: {type_2_error}"
                                   )



def select_mde(
        n_samples: int,
        n_simulations: int,
        cvr: float,
        mde_grid: List[float],
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[float, float]:
    """Select MDE."""

    if not mde_grid:
        raise RuntimeError("Can't find MDE")

    for mde in mde_grid:
        # Implement your solution here

        type_2_error = ab_test(n_simulations, n_samples, cvr, mde, reward_avg, reward_std, alpha)

        if type_2_error <= beta:
            return mde, type_2_error
        if mde == mde_grid[-1] and type_2_error > beta:
            raise RuntimeError(
                "Can't find MDE. "
                f"Last MDE: {mde}, "
                f"last type 2 error: {type_2_error}. "
                "Make sure that the grid is big enough."
            )