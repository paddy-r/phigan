# HR 09/04/25 Iterative proportional fitting (IPF) algorithm for generating synthetic populations
# Adapted from here: https://colab.research.google.com/drive/1QgaNBo3g-HQEgWmpy7i0su5wJxk3A73o

import os
from os.path import dirname as up
import numpy as np
import pandas as pd
import time

DATA_DIR = os.path.join(up(__file__), 'data')

# SIPHER constraints data and related
constraint_dir = os.path.join(DATA_DIR, '_sipher2020_constraints')
constraint_dict = {'age-sex': '2020_est_age_sex.csv',
                   'economic': '2011_census_economic.csv',
                   'ethnicity': '2011_census_ethnicity.csv',
                   'general_health': '2011_census_general_health.csv',
                   'hh_composition': '2011_census_hh_composition.csv',
                   'hh_tenure': '2011_census_hh_tenure.csv',
                   'marital': '2011_census_marital.csv',
                   'qualification': '2011_census_qualification.csv',
                   }
LSOA_COL = 'areacode'
SEX_CATEGORIES = ('m', 'f')
AGE_CATEGORIES = ('16_24', '25_34', '35_49', '50_64', '65_74', '75p')

ETH_GROUPS = {'White': ['WBI', 'WHO',],
              'Black': ['BLA', 'BLC', 'OBL',],
              'Asian': ['IND', 'PAK', 'BAN', 'CHI', 'OAS',],
              'Mixed': ['MIX',],
              'Other': ['OTH',],
              }
DIST_KEYS = ('age_category', 'sex_category')


# HR 18/12/24 To get ethnic supergroup map
def get_ethnicity_category_map():
    _map = {}
    for k, v in ETH_GROUPS.items():
        for g in v:
            _map[g] = k
    return _map


def get_age_intervals():
    age_intervals = [(int(start), int(end) + 1) for start, end in (age_range.split('_') for age_range in AGE_CATEGORIES[:-1])]
    age_intervals += ((75, 200),)
    return age_intervals


def get_age_category_map():
    age_intervals = get_age_intervals()
    age_map = {range(*em): el for el, em in zip(AGE_CATEGORIES, age_intervals)}
    age_map = {age: category for age_range, category in age_map.items() for age in age_range}
    return age_map


# HR 09/04/25 Get SIPHER synthpop constraint data
def get_constraint_data(constraint):
    _file = constraint_dict[constraint]
    return pd.read_csv(os.path.join(constraint_dir, _file)).set_index(LSOA_COL).drop(columns=['total'])


# HR 09/04/25 Convert constraint data to "row"/"col" dictionary form for IPF
def convert_agesex_constraint_data(data):
    data.columns = data.columns.str.split("_", n=1, expand=True)  # Split columns into sex and age for easy indexing
    return data


# HR 10/04/25 Convert economic constraint data to format suitable for IPF
def convert_simple_constraint_data(data):
    _categories = data.columns[~data.columns.isin('total')]
    return data[_categories]


# HR 10/04/25 Get distribution of categorical data by
def get_simple_distribution(data, _var, _keys=DIST_KEYS):
    g = data.groupby(list(_keys))[_var].value_counts(normalize=True).reset_index()
    g = g.pivot(columns=_keys, index=_var, values='proportion').fillna(0)  # fillna accounts for missing groups
    return g.to_dict()


def ipf(seed, row_targets, col_targets, max_iter=100, tol=1e-6):
    start_time = time.time()

    current = {row: cols.copy() for row, cols in seed.items()}
    row_categories = list(row_targets.keys())
    col_categories = list(col_targets.keys())

    for i in range(max_iter):
        # Adjust rows
        for row in row_categories:
            row_sum = sum(current[row].values())
            if row_sum == 0:
                continue
            factor = row_targets[row] / row_sum
            for col in col_categories:
                current[row][col] *= factor

        # Adjust columns
        for col in col_categories:
            col_sum = sum(current[row][col] for row in row_categories)
            if col_sum == 0:
                continue
            factor = col_targets[col] / col_sum
            for row in row_categories:
                current[row][col] *= factor

        # Check convergence
        converged = True
        for row in row_categories:
            if abs(sum(current[row].values()) - row_targets[row]) > tol:
                converged = False
                break
        for col in col_categories:
            if abs(sum(current[row][col] for row in row_categories) - col_targets[col]) > tol:
                converged = False
                break
        if converged:
            break

    end_time = time.time()
    print('Converged in {} iterations; time per iteration: {} mus'.format(i + 1, 1e6*(end_time-start_time)))
    return current


def create_synthetic_population(ipf_counts, _dist):

    for age_group in ipf_counts:
        for sex in ipf_counts[age_group]:
            count = ipf_counts[age_group][sex]
            _age_group = np.array([age_group] * count)
            _sex = np.array([sex] * count)

            # Get probabilities for group
            _lookup = _dist[(age_group, sex)]
            _categories = list(_lookup.keys())
            _probs = list(_lookup.values())

            _cat = np.random.choice(_categories, size=count, p=_probs)

    #         # Generate individuals for this group
    #         for _ in range(int(count)):
    #             # Select income based on probabilities
    #             rand = random.random()
    #             cumulative_prob = 0
    #             income = income_levels[-1]  # default to last category
    #
    #             for i, prob in enumerate(probabilities):
    #                 cumulative_prob += prob
    #                 if rand <= cumulative_prob:
    #                     income = income_levels[i]
    #                     break
    #
    #             synthetic_pop.append({
    #                 'sex': sex,
    #                 'age_group': age_group,
    #                 'ethnicity': ethnicity,
    #             })
    # return
            _pop = np.stack([_age_group, _sex, _cat], axis=1)
            try:
                full_pop = np.concatenate([full_pop, _pop])
            except:
                full_pop = _pop
    return full_pop


if __name__ == "__main__":

    # Example 1: Fake survey data from GAN
    X = 5  # Exponent for number of synthetic individuals to generate
    survey_data = pd.read_csv(os.path.join(DATA_DIR, 'ganpop_1e' + str(X) + '.csv'))
    survey_data['sex_category'] = survey_data['sex'].astype(str).str.lower().str[0]  # Add sex category to match constraint
    survey_data['age_category'] = survey_data['age'].round().astype(int).map(get_age_category_map())  # Add age category to match constraint
    survey_data['ethnicity_category'] = survey_data['ethnicity'].map(get_ethnicity_category_map())  # Add ethnicity category to match constraint

    # Variables to be included in IPF procedure
    target_agesex = convert_agesex_constraint_data(get_constraint_data('age-sex')) * 100
    # target_ethnicity = get_constraint_data('ethnicity')

    # Other non-key variables - to be added after synthpop generation using survey_data distributions
    eth_dist = get_simple_distribution(data=survey_data, _var='ethnicity_category')

    # Example IPF operation
    area = 'E01000001'
    target_raw = target_agesex.loc[[area]].T.reset_index()
    target = {'table': target_raw.groupby(['level_1', 'level_0']).sum().unstack()[area].to_dict(orient='index'),
              'row': target_raw.groupby('level_1')[area].sum().to_dict(),
              'col': target_raw.groupby('level_0')[area].sum().to_dict(),
              'grand_total': target_raw[area].sum(),
              }
    contingency = {'table': survey_data.groupby(['age_category', 'sex_category']).size().unstack().to_dict(orient='index'),
                   'row_totals': survey_data.groupby(['age_category']).size().to_dict(),
                   'col_totals': survey_data.groupby(['sex_category']).size().to_dict(),
                   'grand_total': len(survey_data),
                   }

    result = ipf(seed=contingency['table'], row_targets=target['row'], col_targets=target['col'])

    synthpop = create_synthetic_population(target['table'], eth_dist)
    df = pd.DataFrame(synthpop)
    df.columns = list(DIST_KEYS) + ['ethnicity_category']


    # Create standardised SIPHER 2020 constraints data
    targets = {}
    agesex = get_constraint_data('age-sex')
    targets['age'] = agesex.T.groupby(['_'.join(s.split('_')[1:]) for s in agesex.T.index.values]).sum().T
    targets['sex'] = agesex.T.groupby([s.split('_')[0] for s in agesex.T.index.values]).sum().T
    for t in ('age-sex', 'economic', 'ethnicity', 'general_health', 'hh_composition', 'hh_tenure', 'marital', 'qualification'):
        targets[t] = get_constraint_data(t)

    outpath = os.path.join(constraint_dir, 'constraints_standardised')
    for target, data in targets.items():
        outfile = target + '.csv'
        outfull = os.path.join(outpath, outfile)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        data.to_csv(outfull)
