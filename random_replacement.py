# HR 08/04/25 Toy random replacement code for generating synthetic populations, constrained on arbitrary variables
# Adapted from here: https://github.com/Leeds-MRG/Minos/blob/full-retention-for-synthpop/minos/modules/replenishment.py
import sys
import numpy as np
import pandas as pd
import os
from os.path import dirname as up

DATA_DIR = os.path.join(up(__file__), 'data')


# HR 09/02/25 Return Euclidean distance between two vectors
def euclidean(v1, v2):
    d = np.sqrt(np.sum((v1 - v2) ** 2))
    return d


# HR 09/02/25 Objective function for simulated annealing function - used as measure of convergence
def objective_function(df, target_dict):
    PENALTY_VALUE = 10.0  # Setting this >0 avoids samples with empty categories being produced

    obj = 0.0
    for v, t in target_dict.items():
        if isinstance(t, (int, float)):  # For int/float
            m = df[v].mean()
            new_val = euclidean(m, t)
        elif isinstance(t, dict):  # For categoricals

            vc = df[v].value_counts(normalize=True)

            # Must check all categories in target are present; if not, add zero value to avoid ValueError
            if set(t) != set(df[v]):
                not_present = set(t) - set(df[v])
                for _np in not_present:
                    vc.loc[_np] = PENALTY_VALUE

            vec = np.array(vc.sort_index())
            t_sorted = ([v for (k, v) in sorted(t.items())])

            new_val = euclidean(vec, t_sorted)
        else:
            new_val = PENALTY_VALUE
        obj += new_val
    return obj


def sample_with_constraints(df,
                            target_dict,
                            frac=0.1,
                            n=None,
                            delta_threshold=0.002,  # Convergence threshold
                            subfrac=0.005,  # Relative size of subsample to replace
                            T_0=1000.0,  # Initial temperature
                            alpha=0.99,  # Cooling rate
                            ):
    """
    Returns a fractional sample of the input dataframe with a set of values close to the target set.
    Uses simulated annealing to find the sample.

    Parameters:
    df (pandas.DataFrame): The input dataframe
    target_dict (dict): The target set of values
    frac (float): The size of the sample to be returned, expressed as a fraction of the input dataframe

    Returns:
    pandas.DataFrame: A fractional sample of the input dataframe with a mean value close to the target values
    """
    # Get size of sample to create -> this prioritises n if it is specified
    if n is not None and isinstance(n, (int, float,)):
        frac = n / len(df)

    # Initialise variables
    oversample = frac > 1.0  # Only allow for duplicates per sample if requested size bigger than repl source pop
    current_sample = df.sample(frac=frac, replace=oversample)
    current_obj = objective_function(current_sample, target_dict)  # Objective of current sample
    T = T_0

    # Run simulated annealing loop
    i = 0

    # while T > 1.0:
    while current_obj > delta_threshold:

        # 1. Get subsample to be used as replacement
        n_replace = int(subfrac * frac * len(df))
        to_replace = df.sample(n=n_replace)

        # 2. Replace random rows in current sample with subsample
        new_sample = current_sample.sample(frac=1)[:-n_replace]  # Shuffle then drop last n rows
        new_sample = pd.concat([new_sample, to_replace])

        # 3. Calculate objective of proposed sample
        new_obj = objective_function(new_sample, target_dict)
        diff = new_obj - current_obj

        # 4. If proposed sample better than current sample, keep it; otherwise discard
        # Accept or reject the new sample based on the Metropolis criterion
        # if diff < 0 or np.exp(-diff / T) > np.random.rand():
        if diff < 0:
            current_sample = new_sample
            current_obj = new_obj

        # Cool down the system
        T *= alpha
        sys.stdout.write('\rIteration no. {} (obj: {:.6f}), N = {}'.format(i, current_obj, len(current_sample)))

        # # Check if the current sample is close enough to the target mean
        # if abs(current_obj - target) < delta_threshold:
        #     break

        i += 1

    print('\r')
    return current_sample, current_obj


if __name__ == "__main__":
    # Example 1: Fake areas with some simple constraints, using Minos synthpop
    # source = pd.read_csv(os.path.join(DATA_DIR, '2019_US_cohort.csv'))

    # Example 2: Fake areas with source from GAN, 1eX individuals
    X = 5  # Exponent for number of synthetic individuals to generate
    source = pd.read_csv(os.path.join(DATA_DIR, 'ganpop_1e' + str(X) + '.csv'))

    num_areas = 10
    area_pop = 1500

    # Toy synthpop generation with (mean) age and ethnicity constraints using random sampling
    perturbation = 0.01  # How much to perturb ethnicity distribution
    eths = source['ethnicity'].unique()
    # for i in range(num_areas):

    # Perturb ethnicity distribution
    eth_con = source.ethnicity.value_counts(normalize=True)
    eth_con = abs(eth_con + np.random.normal(0, perturbation, size=len(eth_con)))
    eth_con /= eth_con.sum()  # Normalise
    eth_con = dict(zip(eths, eth_con))  # Distribution-type constraint, normalised
    age_con = np.random.randint(35, 45)  # Mean-type constraint

    constraints = {'ethnicity': eth_con,
                   'age': age_con,
                   }

    synthpop, obj = sample_with_constraints(source, target_dict=constraints, n=area_pop, subfrac=0.005,
                                            delta_threshold=0.02)
