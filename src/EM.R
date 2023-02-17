


l2norm <- function(x) {x/sqrt(sum(x^2))}

log.L <- function(x, Pi, G, param){
  Z <- length(Pi)
  a1 <- matrix(0, Z, N)
  for (z in 1:Z){
    a1[z,] <- Pi[z]*kent.density(x, G[z,,], param[z,])
  }
  likelihoods <- colSums(a1)
  return(sum(log(likelihoods)))
}

Q.dir <- function(x, tau, G, param){
  Z <- ncol(tau)
  a1 <- 0
  for (z in 1:Z) {
    a1 <- a1 + sum(tau[,z]*kent.density(x, G[z,,], param[z,], logden = TRUE))
  }
  return(a1)
}

E.step <- function(x, Pi, G, param){
  N <- nrow(x)
  Z <- length(Pi)
  tau <- matrix(0, N, Z)
  for (z in 1:Z){
    tau[,z] <- Pi[z]*kent.density(x, G[z,,], param[z,])
  }
  # sum.tau <- apply(tau,1,sum)
  # tau <- sweep(tau,1,FUN="/",STAT=sum.tau)
  row_sums <- rowSums(tau)
  tau <- diag(1/row_sums) %*% tau
  return(tau)
}
optim_fn <- function(parameters, x, tau){
  Z <- ncol(tau)
  G <- angles2G(parameters[1:(Z*3)])
  param <- array(data=parameters[-(1:(Z*3))], dim=c(Z, 3)) # kappa and beta/kappa
  param[,2] <- param[,1] * param[,2] # kappa and beta
  return(-Q.dir(x, tau, G, param))
}

M.step <- function(x, tau, G0, param0){
  Z <- ncol(tau)
  Pi <- apply(tau, 2, mean)
  angles0 <- G2angles(G0)
  parameters0 <- c(as.vector(angles0), as.vector(param0))
  result <- optim(parameters0, optim_fn, gr = NULL, 
                  method='L-BFGS-B',
                  lower=c(rep(0, Z*3), rep(0, Z), rep(0, Z), rep(-Inf, Z)), 
                  upper=c(rep(pi, 2*Z), rep(2*pi, Z), rep(Inf, Z), rep(0.5 - 1e-5, Z), rep(Inf, Z)), 
                  x=x, tau=tau)
  G <- angles2G(result$par[1:(Z*3)])
  param <- array(data=result$par[-(1:(Z*3))], dim=c(Z, 3)) # kappa and beta/kappa
  param[,2] <- param[,1] * param[,2] # kappa and beta
  return(list(Pi = Pi, G = G, param = param))
}




# EM algorithm
# Initialization
x <- X_1
tol <- 0.0002
BIC_kent_no_conta_rock1 <- c()
A_kent_no_conta_rock1 <- list()
id_kent_no_conta_rock1 <- list()
tau_kent_no_conta_rock1 <- list()

for (Z in 2:12) {
 set.seed(Z)
 N <- nrow(x)
 id.km <- skmeans(x,Z)$cluster
 tau.km <- matrix(0,N,Z)
 #vm <- mix.vmf(x, Z, 20)
 #id.km <- vm$pred
 for (z in 1:Z){
   tau.km[id.km == z,z] <- 1
 }


 param0 <- array(0, c(Z, 3))
 G0 <- array(0, c(Z, 3, 3))
 bic <- c()
 for(i in 1:Z){
   result <- kent.mle(x[tau.km[,i]==1,])
   G0[i,,] <- result$G
   param0[i,] <- result$param
 }
 A <- M.step(x, tau.km, G0, param0)

 ll.old <- -Inf
 s <- 0
 repeat{
   s <- s + 1
   tau <- E.step(x, A$Pi, A$G, A$param)
   A <- M.step(x, tau, A$G, A$param)
   # A$G <- aperm(apply(A$G, c(1,3), l2norm), perm=c(2, 1, 3))
   ll <- log.L(x, A$Pi, A$G, A$param)
   tau <- E.step(x, A$Pi, A$G, A$param)
   A <- M.step(x, tau, A$G, A$param)
  # A$G <- aperm(apply(A$G, c(1,3), l2norm), perm=c(2, 1, 3))
   ll <- log.L(x, A$Pi, A$G, A$param)
   cat("Iteration", s, "logL =", ll, "\n")
   if ((ll - ll.old) / abs(ll) < tol) break
   ll.old <- ll
 }
 M <- 3 * Z - 1
 bic <- -2 * ll + M * log(N)
 cat("BIC =", bic, "\n")
 BIC_kent_no_conta_rock1[Z] <- bic
 A_kent_no_conta_rock1[[Z]] <- A
 tau_kent_no_conta_rock1[[Z]] <- tau
 id_kent_no_conta_rock1[[Z]] <- apply(tau, 1, which.max)
}




id.dir <- apply(tau, 1, which.max)
# vonMises mixture model
id.vonMF <- predict(movMF(x, 3, nruns=20))
corresponding_true_clusters.km <- apply(table(id.true, id.km), 1, which.max)
corresponding_true_clusters.dir <- apply(table(id.true, id.dir), 1, which.max)
corresponding_true_clusters.vonMF <- apply(table(id.true, id.vonMF), 1, which.max)

id.km1 <- corresponding_true_clusters.km[id.km]
id.dir1 <- corresponding_true_clusters.dir[id.dir]
id.vonMF1 <- corresponding_true_clusters.vonMF[id.vonMF]

table(id.true, id.dir1)
table(id.true, id.km1)
table(id.true, id.vonMF1)

ClassProp(id.true, id.dir)
ClassProp(id.true, id.km)
ClassProp(id.true, id.vonMF)

RandIndex(id.true, id.dir)
RandIndex(id.true, id.km)
RandIndex(id.true, id.vonMF)






















