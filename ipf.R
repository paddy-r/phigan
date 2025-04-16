library(purrr)
# library(ipfp)
library(mipfp)


normaliseVector <- function(v, ref){
  # For standardising marginals/constraints so total counts match master
  names_ <- names(v)
  v <- v * (sum(ref) / sum(v))
  names(v) <- names_
  v
}


int_trs <- function(x){
  # For generalisation purpose, x becomes a vector
  xv <- as.vector(x) # allows trs to work on matrices
  xint <- floor(xv) # integer part of the weight
  print(xv)
  r <- xv - xint # decimal part of the weight
  print('r')
  print(r)
  def <- round(sum(r)) # the deficit population
  print(def)
  # the weights be 'topped up' (+ 1 applied)
  topup <- sample(length(x), size = def, prob = r)
  print('topup:')
  print(topup)
  xint[topup] <- xint[topup] + 1
  dim(xint) <- dim(x)
  dimnames(xint) <- dimnames(x)
  xint
}

### HR 15/04/25 To convert from contingency table to dataframe of individuals
### Taken from R Cookbook here: http://www.cookbook-r.com/Manipulating_data/Converting_between_data_frames_and_contingency_tables/
# Convert from data frame of counts to data frame of cases.
# `countcol` is the name of the column containing the counts
countsToCases <- function(x, countcol = "Freq") {
  # Get the row indices to pull from x
  idx <- rep.int(seq_len(nrow(x)), x[[countcol]])
  
  # Drop count column
  x[[countcol]] <- NULL
  
  # Get the rows from x
  x[idx, ]
}


constraints.path <- file.path(here::here(), 'data', '_sipher2020_constraints', 'constraints_standardised')
files <- list.files(path=constraints.path, pattern="*.csv")
area_col <- 'areacode'
area <- 'E01000001'

targets <- c()
constraints <- c()
names_ <- c()
for (file in files) {
  t <- tools::file_path_sans_ext(file)
  targets[[t]] <- read.csv(file.path(constraints.path, file), check.names=FALSE, row.names=area_col)
  # targets[[t]] <- read.csv(file.path(constraints.path, file), row.names=area_col)
  constraints[[t]] <- unlist(targets[[t]][area,])
  names_[[t]] <- unlist(names(constraints[[t]]))
  print(t)
}

# for (i in seq_along(targets)) {
#   print(paste(i, names(targets)[i]))
# }


# Test 1: Running with three constraints with IPFP
# TO DO

# Test 2: Running with two constraints with MIPFP
agesex.cons <- unlist(targets[['age-sex']][area,])
eth.cons <- unlist(targets$ethnicity[area,])

# Rescale to match total for agesex constraint
# names_ <- names(eth.cons)
# eth.cons <- eth.cons * (sum(agesex.cons) / sum(eth.cons))
# # eth.cons <- int_trs(eth.cons)
# names(eth.cons) <- names_
eth.cons <- normaliseVector(eth.cons, agesex.cons)

names <- list(names(agesex.cons), names(eth.cons))
seed <- array(10, dim=c(length(agesex.cons), length(eth.cons)), dimnames=names)
descript <- list(1, 2)
target <- list(agesex.cons, eth.cons)
              
result2 <- Ipfp(seed, descript, target, iter=50, print=TRUE, tol=1e-6)
result2$x.hat
sum(result2$x.hat)

# Test 2a: Running with three constraints with MIPFP
age.cons <- unlist(targets$age[area,])
sex.cons <- unlist(targets$sex[area,])
eth.cons <- unlist(targets$ethnicity[area,])

names <- list(names(age.cons), names(sex.cons), names(eth.cons))
seed <- array(10, dim=c(length(age.cons), length(sex.cons), length(eth.cons)), dimnames=names)
descript <- list(1, 2, 3)
target <- list(age.cons, sex.cons, eth.cons)

result4 <- Ipfp(seed, descript, target, iter=50, print=TRUE, tol=1e-6)
result4$x.hat
sum(result4$x.hat)

# Test 2b: Running with four constraints with MIPFP
age.cons <- unlist(targets$age[area,])
sex.cons <- unlist(targets$sex[area,])
eth.cons <- unlist(targets$ethnicity[area,])
hh_tenure.cons <- unlist(targets$hh_tenure[area,])

names <- list(names(age.cons), names(sex.cons), names(eth.cons), names(hh_tenure.cons))
seed <- array(10, dim=c(length(age.cons), length(sex.cons), length(eth.cons), length(hh_tenure.cons)), dimnames=names)
descript <- list(1, 2, 3, 4)
target <- list(age.cons, sex.cons, eth.cons, hh_tenure.cons)

resultN <- Ipfp(seed, descript, target, iter=50, print=TRUE, tol=1e-6)
resultN$x.hat
sum(resultN$x.hat)

# Test 3: Running with N constraints with MIPFP
names.auto <- names_
seed.auto <- array(10, dim=unname(lengths(constraints)), dimnames=names.auto)
descript.auto <- seq(length(constraints))
target.auto <- unname(constraints)

result.auto <- Ipfp(seed.auto, descript.auto, target.auto, iter=50, print=TRUE, tol=1e-6)
result.auto$x.hat
