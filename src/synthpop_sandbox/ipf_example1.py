# HR 25/07/25 Standardised IPF example: 3D solution with 2D constraints using IPFN and PyIPF (identical solutions)
# Adapted from here: https://github.com/Dirguis/ipfn/issues/25
# Also done in R (via MIPFP) - see R script with same name

import numpy as np
import ipfn
import pyipf


''' HR 24/07/25 To prepare constraints in xyz format (i.e. xy, yz, zx) for use by PyIPF,
    which needs them in yz, xz, xy, which can be computed from np.sum(axis=X), where X = {0, 1, 2},
    roll_value allows for rotating/transposing dimensions to see whether order affects result
'''
def format_constraints(dims_xyz, m_xyz, constraints_xyz, roll_value=0):

    dims = np.roll(dims_xyz, shift=roll_value+2)
    m = np.transpose(m_xyz, axes=np.roll(list(range(m_xyz.ndim)), shift=roll_value))
    constraints = constraints_xyz[-(roll_value+2):] + constraints_xyz[:-(roll_value+2)]
    constraints[1] = constraints[1].T

    return dims, m, constraints


# HR 24/07/25 Example for testing with Python/R and whatever packages are available
# Adapted from here: https://github.com/Dirguis/ipfn/issues/25
m = np.ones((2, 4, 3))
xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])  # xy 2x3
xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])  # yz, 4x3
xipk = np.array([[22, 20, 10], [13, 20, 15]])  # xz, 2x3


# TEST 1: Using IPFN, repo here: https://github.com/Dirguis/ipfn
aggregates = [xijp, xpjk, xipk]
dimensions = [[0, 1], [1, 2], [0, 2]]

model = ipfn.ipfn.ipfn(m.copy(), aggregates, dimensions, verbose=2)  # Must copy m as it is mutated
result = model.iteration()

print('Final convergence value: {}'.format(result[-1].conv.iloc[-1]))
print('Convergence sequence:\n', result[-1])

print('\nTarget sum (xy constraint):', xijp.sum())
print('IPF gives:', result[0].sum(), '\n')

print('xy constraint vs. computed:')
print(aggregates[0], '\n\n', result[0].sum(axis=2), '\n')

print('Value at [0, 0] (xy constraint): {}'.format(xijp[0,0]))
print('Value at [0, 0] (result):', result[0][0, 0, :].sum())

print('\n', result[0])  # Solution


# TEST 2: Using PyIPF, repo here: https://github.com/MutaharChalmers/pyipf
# Adapted from Jupyter notebook buried here: https://github.com/MutaharChalmers/pyipf/blob/main/docs/README.ipynb
aggregates = [xpjk, xipk, xijp]  # Note different order of constraints to IPFN
result = pyipf.ipf(m.copy(), aggregates, max_itr=100, pbar=True)

print('\nTarget sum (xy constraint):', xijp.sum())
print('IPF gives:', result.sum(), '\n')

print('xy constraint vs. computed:')
print(aggregates[2], '\n\n', result.sum(axis=2), '\n')

print('Value at [0, 0] (xy constraint): {}'.format(xijp[0,0]))
print('Value at [0, 0] (result):', result[0, 0, :].sum())

print('\n', result)  # Solution
