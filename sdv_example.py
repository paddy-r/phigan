# HR 05/06/25 To create example to test basic SDV synthetic data creation

import os
import pandas as pd
import ipf

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
# from sdv.evaluation.multi_table import run_diagnostic, evaluate_quality
# from sdv.evaluation.multi_table import get_column_plot

DATA_DIR = ipf.DATA_DIR


# data = pd.read_csv('my_data_file.csv')
data = pd.read_csv(os.path.join(DATA_DIR, '2019_US_cohort.csv'))  # Basic Minos synthpop
data = data.head(round(0.01*len(data)))
to_drop = ['LSOA11CD', 'pidp', 'weight', 'hidp', 'birth_year', 'child_ages', 'child_ages_ind', 'hh_int_y', 'hh_int_m', 'time']
data.drop(columns=to_drop, inplace=True)
metadata = Metadata.detect_from_dataframe(data)


# Example 1: US 2019 data with a Gaussian copula
# Adapted from here: https://docs.sdv.dev/sdv
synthesizer1 = GaussianCopulaSynthesizer(metadata)
synthesizer1.fit(data)
sd1 = synthesizer1.sample(num_rows=10000)


# Example 2: US 2019 data with a CTGAN
# Adapted from here: https://colab.research.google.com/drive/15iom9fO8j_gHg4-NlGkzWF5thMWStXwv
synthesizer2 = CTGANSynthesizer(metadata)
synthesizer2.fit(data)
sd2 = synthesizer2.sample(num_rows=10000)


# # POST - looking at some statistics
# # Adapted from here: https://docs.sdv.dev/sdv/multi-table-data/evaluation
#
# # 1. perform basic validity checks
s = sd2
diagnostic = run_diagnostic(data, s, metadata)

# 2. measure the statistical similarity
quality_report = evaluate_quality(data, s, metadata)

# 3. plot the data
fig = get_column_plot(
    real_data=data,
    synthetic_data=s,
    metadata=metadata,
    column_name='ethnicity',
)

fig.show()
