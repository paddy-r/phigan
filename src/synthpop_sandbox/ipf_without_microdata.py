import os
import time
import itertools
import numpy as np
import pandas as pd
from ipfn import ipfn
import ipf_scratch
import trs

DATA_DIR = ipf_scratch.DATA_DIR

# Example from IPFN Github readme
# Copied from here: https://github.com/Dirguis/ipfn
# # m = np.array([[30, 40, 20, 10], [35, 50, 100, 75], [30, 80, 70, 120], [50, 30, 40, 20]])
# m = np.ones([4, 4])
#
# a = np.array([150, 300, 400, 150])
# b = np.array([200, 300, 400, 100])
#
# aggregates = [a, b]
# dimensions = [[0], [1]]
#
# IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-6)
# result = IPF.iteration()
# print(result)


# HR 16/04/25 Normalize constraint to sum to reference vector/matrix
# To ensure IPF constraints have same sum
def normalise_constraint(vc, ref):
    return vc * ref.sum() / vc.sum()


# HR 24/04/25 To create synthetic population
# i.e. create columns with values corresponding to indices in IPF solution
def create_synthetic_population(result, category_keys, categories=None):
    # Just label with integers if no category labels given
    if categories is None:
        categories = [str(el) for el in range(len(category_keys))]

    # Flatten IPF result and category keys, then combine into list
    result_seq = result.flatten()
    key_seq = [el for el in itertools.product(*category_keys)]
    freq = [el for el in zip(key_seq, result_seq.astype(int))]

    # Create population by repeating rows N times per cell in IPF contingency table
    # pop = pd.DataFrame(freq, columns=categories)
    pop = pd.DataFrame(freq)
    pop = pop.reindex(pop.index.repeat(pop[pop.columns[-1]])).drop(columns=pop.columns[-1]).reset_index(drop=True)  # Last column will always be frequency

    # Expand categories column from tuple into individual variables
    pop[categories] = pop[pop.columns[-1]].to_list()  # Again can just use last column as only one column present
    pop.drop(columns=pop.columns[0], inplace=True)
    return pop


# HR 24/04/25 Expand age-sex column into two separate columns
def expand_agesex(pop, agesex_col='agesex'):
    popexp = pop.copy()
    popexp[['sex', 'age']] = popexp[agesex_col].str.split('_', n=1, expand=True)
    popexp = popexp.drop(columns=agesex_col)
    return popexp


if __name__ == "__main__":
    # Get some survey data to use as seed
    survey_data = pd.read_csv(os.path.join(DATA_DIR, '2019_US_cohort.csv'))  # Basic Minos synthpop
    # survey_data = pd.read_csv(os.path.join(DATA_DIR, 'ganpop_1e5.csv'))  # GAN 1e6 synthpop, generated from Minos synthpop above

    survey_data['sex_category'] = survey_data['sex'].astype(str).str.lower().str[0]  # Add sex category to match constraint
    survey_data['age_category'] = survey_data['age'].round().astype(int).map(ipf_scratch.get_age_category_map())  # Add age category to match constraint
    survey_data['ethnicity_category'] = survey_data['ethnicity'].map(ipf_scratch.get_ethnicity_category_map()).str.lower()  # Add ethnicity category to match constraint
    survey_data['agesex_category'] = survey_data['sex_category'] + '_' + survey_data['age_category']  # Combine into single agesex column to match constraint

    # Testing - want to reproduce results from R MIPFP code in ipf.R using SIPHER synthpop 2020 constraints
    area = 'E01000001'

    # More testing - loop over N LSOAs and get timings...
    constraint_ref = 'age-sex'
    N = 10
    lsoas = ipf_scratch.get_constraint_data(constraint_ref).index[:N].to_list()  # Get first N LSOAs from (e.g.) age-sex constraint file

    constraint_vars = list(ipf_scratch.constraint_dict.keys())
    # constraint_vars = [list(ipf.constraint_dict.keys())[el] for el in [0, 2]]  # Select particular variables
    constraint_data = {c: ipf_scratch.get_constraint_data(c) for c in constraint_vars}

    # Get seed - same for all LSOAs here, although can calculate for each
    dims = [v.shape[1] for v in constraint_data.values()]

    # Construct seed data matrix
    seed = np.full(shape=dims, fill_value=10)  # To match R example 2
    # seed = np.random.rand(*dims)  # Alternative seed - random numbers
    # seed = pd.crosstab(survey_data['agesex_category'], survey_data['ethnicity_category']).to_numpy()  # Alternative seed - crosstab from synthpop data

    # Create empty dataframe to append to for each LSOA
    full_pop = pd.DataFrame()

    # Timings
    t0 = []  # Reference point
    t1 = []  # To do IPF
    t2 = []  # To create synthpop in Pandas
    t3 = []  # To add additional variable from survey data

    print('')
    for i, area in enumerate(lsoas):
        # start = time.time()
        t0.append(time.time())

        print("\rCalculating IPF solution for LSOA {} ({} of {})".format(area, i+1, N))
        aggregates = []
        cat_keys = []
        col_names = []

        for c in constraint_vars:
            raw = constraint_data[c].loc[area].sort_index()
            con = raw.to_numpy()
            if c == constraint_ref:
                ref_con = con
                ref_sum = con.sum()
            else:
                if con.sum() != ref_sum:
                    con = normalise_constraint(con, ref_con)
                    # con = trs.trs(con)
            key = raw.index
            aggregates.append(con)
            cat_keys.append(key)
            col_names.append(c)


        # dimensions = [[0], [1], [2]]
        dimensions = [[i] for i in range(len(aggregates))]
        # seed = np.full(shape=[len(el) for el in aggregates], fill_value=10)  # To match R example 2
        # seed = np.random.rand(*[len(el) for el in aggregates])  # Alternative seed - random numbers
        # seed = pd.crosstab(survey_data['agesex_category'], survey_data['ethnicity_category']).to_numpy()  # Alternative seed - crosstab from synthpop data

        IPF = ipfn.ipfn(seed, aggregates, dimensions, convergence_rate=1e-6)
        result = IPF.iteration()
        t = trs.trs(result)  # Integerised result


        dt1 = time.time()
        t1.append(dt1)


        # Create dataframe of population
        pop = create_synthetic_population(t, cat_keys, categories=col_names)


        dt2 = time.time()
        t2.append(dt2)


        # Adding additional or non-constraint variables, e.g. from SIPHER synthpop: nkids_ind
        var_to_add = 'nkids_ind'  # Variable to add to population
        ref_vars = ['agesex_category', 'ethnicity_category']  # Reference variables from which to draw distribution
        ct = pd.crosstab([survey_data[v] for v in ref_vars], survey_data[var_to_add], normalize='index').to_dict(orient='index')  # Create dictionary for sampling

        # Sample from distribution for each combination of reference variables (quicker than going row by row)
        pop_cols = ['age-sex', 'ethnicity']  # These are corresponding columns in synthetic population - categories MUST match those in crosstab
        pop['comb'] = pop[pop_cols].apply(tuple, axis=1)
        for comb, dist in ct.items():
            # Get subframe of individuals with that combination
            sub = pop.loc[pop['comb'] == comb]
            if len(sub) == 0:
                continue

            # Sample from distribution for that combination
            uniques = list(dist.keys())
            probs = list(dist.values())
            sample_values = np.random.choice(uniques, size=len(sub), replace=True, p=probs)

            # Paste onto subset
            pop.loc[sub.index, var_to_add] = sample_values
        pop.drop(columns=['comb'], inplace=True)
        pop.insert(0, 'areacode', area)
        full_pop = pd.concat([full_pop, pop])


        dt3 = time.time()
        t3.append(dt3)


        # end = time.time()
        # elapsed = end - start
        # t1.append(elapsed)

    # Metrics for speed/size
    n_areas = 41_729
    fileout = 'full_pop.csv'
    full_pop.to_csv(fileout)
    outsize = os.path.getsize(fileout) / 1024**2
    dt1 = (sum(t1) - sum(t0))
    dt2 = (sum(t2) - sum(t1))
    dt3 = (sum(t3) - sum(t2))
    print("Elapsed per LSOA (t1): {}s, for all GB: {}h".format(dt1 / N, n_areas * dt1 / (N * 60 * 60)))
    print("Elapsed per LSOA (t2): {}s, for all GB: {}h".format(dt2 / N, n_areas * dt2 / (N * 60 * 60)))
    print("Elapsed per LSOA (t3): {}s, for all GB: {}h".format(dt3 / N, n_areas * dt3 / (N * 60 * 60)))
    print("Size of output file: {:.2f}MB".format(outsize))
    print("Size for full GB: {:.2f}MB".format(outsize * n_areas / N))


    # HR 29/04/25 Four cases to look at effect of seed data - any LSOA will do
    agesex_raw = ipf_scratch.convert_agesex_constraint_data(ipf_scratch.get_constraint_data('age-sex')).loc[area].unstack()
    agesex_con = agesex_raw.to_numpy()

    age_cats = agesex_raw.columns
    sex_cats = agesex_raw.index

    sex_con = agesex_con.sum(axis=1)
    age_con = agesex_con.sum(axis=0)
    aggregates = [sex_con, age_con]
    dimensions = [[0], [1]]

    # Case 1: Cross-tabbed age-sex constraints as "ground truth"
    # seed = agesex_con

    # Case 2: SIPHER synthpop (2020) population (result, NOT constraints used for it)
    # seed = pd.crosstab(survey_data['sex_category'], survey_data['age_category']).to_numpy()

    # Case 3: Random values
    # seed = np.random.rand(*[len(el) for el in aggregates])  # Alternative seed - random numbers

    # Case 4: "Uniform prior", i.e. single value
    seed = np.full(shape=[len(el) for el in aggregates], fill_value=10)  # To match R example 2

    # Run IPF
    IPF = ipfn.ipfn(seed, aggregates, dimensions, convergence_rate=1e-6)
    experiment2 = IPF.iteration()
    try:
        asints = trs.trs(experiment2)  # Integerised result
    except:
        print('TRS failed, probably because there are no floats in there...')
        asints = experiment2
    df = pd.DataFrame(asints.astype(int), index=sex_cats, columns=age_cats)
