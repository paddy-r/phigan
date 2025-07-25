library(mipfp)

# HR 23/07/25 Adapted version of Python IPFN example in readme, for 2D constraints
# See here: https://github.com/Dirguis/ipfn
# Adapted from example here: https://github.com/Dirguis/ipfn/issues/25
# Also done in Python (via IPFN and PyIPF) - see Python script with same name

seed <- array(1, c(2, 4, 3))

xijp <- array(c(c(9, 17, 19, 7), c(11, 13, 16, 8)), c(4, 2))  # xy, 2x4
xpjk <- array(c(c(7, 9, 4), c(8, 12, 10), c(15, 12, 8), c(5, 7, 3)), c(3, 4))  # yz, 4x3
xipk <- array(c(c(22, 20, 10), c(13, 20, 15)), c(3, 2))  # zx, 2x3

target <- list(xijp, xpjk, xipk)
descript <- list(c(2, 1), c(3, 2), c(3, 1))

result <- Ipfp(seed, descript, target, iter=1000, print=TRUE, tol=1e-6)

# Can then get sums for each dimension - i.e. sums over faces - to compare to constraints like this:
i <- 0
for(constraint in target){
  i <- i+1
  print(target[i])
  print(margin.table(result$x.hat, margin=descript[[i]]))

}

print(result$x.hat
)