# HR 15/04/25 Python version of R function int_trs
# Integerises list of non-integers according to "truncate, repicate, sample" logic
# Paper is here: https://www.sciencedirect.com/science/article/pii/S0198971513000240
# R version (with examples) is here: https://spatial-microsim-book.robinlovelace.net/smsimr#sintegerisation

# int_trs <- function(x){
#   # For generalisation purpose, x becomes a vector
#   xv <- as.vector(x) # allows trs to work on matrices
#   xint <- floor(xv) # integer part of the weight
#   r <- xv - xint # decimal part of the weight
#   def <- round(sum(r)) # the deficit population
#   # the weights be 'topped up' (+ 1 applied)
#   topup <- sample(length(x), size = def, prob = r)
#   xint[topup] <- xint[topup] + 1
#   dim(xint) <- dim(x)
#   dimnames(xint) <- dimnames(x)
#   xint
# }

import numpy as np

# HR 16/04/25 TRS algorithm, direct adaptation of R version (int_trs)
def trs(x):
    xv = x.reshape(-1)  # Flatten to 1D array to work on matrices of arbitrary dimensions
    xint = np.floor(xv)  # Integer part
    r = xv - xint  # Decimal part
    deficit = round(r.sum())  # Deficit population
    topup = np.random.choice(len(xv), size=deficit, p=r/r.sum(), replace=False)  # Must NOT allow replacement, as in R
    xint[topup] += 1
    xint = xint.reshape(x.shape)  # Reshape to original dimensions
    return xint


# HR 16/04/25 Add Gaussian noise, ensure all values positive and rescale to match original sum
# For producing noisy contingency tables to test TRS algorithm on
def jumble(x, noise_level=0.1):
    xv = x.reshape(-1)  # Flatten to 1D array to work on matrices of arbitrary dimensions
    noise = np.random.normal(1, noise_level * xv.mean(), len(xv))
    xnoise = xv + abs(noise)  # Add positive noise
    xnoise *= xv.sum() / xnoise.sum()  # Rescale so sum is same as input
    result = xnoise.reshape(x.shape)  # Reshape to original dimensions
    return result


if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Original data to be recovered

    # Example 1: Higher noise level, vector data
    noise_level = 0.2
    noisy1 = jumble(data, noise_level)
    result1 = trs(noisy1)
    print('Example 2: Noise level {}, dimensions {}'.format(noise_level, noisy1.shape))
    print('Input sum: {}, output sum: {}'.format(noisy1.sum(), result1.sum()))

    # Example 2: Lower noise level, 2D data
    noise_level = 0.001
    data2 = np.array(data).reshape(2, 6)
    noisy2 = jumble(data2, noise_level)
    result2 = trs(noisy2)
    print('Example 1: Noise level {}, dimensions {}'.format(noise_level, noisy2.shape))
    print('Input sum: {}, output sum: {}'.format(noisy2.sum(), result2.sum()))
