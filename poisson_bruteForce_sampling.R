#example of sampling poisson
lambda <- 5
max_count <- 3 * lambda
x <- 1:max_count
prior_x1 <- dpois(1:max_count,lambda = lambda)
prior_x2 <- dpois(1:max_count,lambda = lambda)
obs_y <- dpois(1:(4*max_count),lambda = (4*lambda)) 
  
joint <- matrix(nrow = max_count, ncol = max_count)
for (i in 1:max_count){
  for (j in 1:max_count) {
    lik = obs_y[i+j]
    joint[i,j] <- prior_x1[i] * prior_x2[j] * lik
  }
}

postX1 <- colSums(joint)
postX2 <- rowSums(joint)
