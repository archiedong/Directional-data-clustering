

library(Directional)
library(ggplot2)
library(MixSim)
library(skmeans)
library(data.table)
library(plotly)
library(abind)
library(movMF)
library(numDeriv)
library(optimx)
library(pracma)

PARAMS_PER_CLUSTER = 6
EPSILON <- 1e-6

# data generation by envelope method
my_rkent <- function(n, k, m, b, major, minor){
 N <- 10000
 n_total <- 0
 MAX_ITERS <- 10000
 n_iters <- 0
 output_list <- list()
 while(n_total<n){
   n_iters <- n_iters + 1
   G <- cbind(m, major, minor)
   param <- c(k, b)
   lat <- runif(N, min=-90, max=90)
   lon <- runif(N, min=-180, max=180)
   r <- cos(lat/180*pi)
   x <- r*cos(lon/180*pi)
   y <- r*sin(lon/180*pi)
   z <- sin(lat/180*pi)
   X <- cbind(x, y, z)
   ds <- kent.density(X, G, param)
   max_d <- max(ds)
   ro <- runif(N, min=0, max=max_d)
   iter_output <- X[ro<ds, ]
   output_list[[n_iters]] <- iter_output
   n_total <- n_total + nrow(iter_output)
   if(n_iters > MAX_ITERS) {
     stop("Reached maximum iterations before collecting enough data points.")
   }
 }
 return(do.call(rbind, output_list)[1:n,])
}


angles2G <- function(angles){
  Z <- length(angles)/3
  G<- array(rep(0, Z*3*3), dim = c(Z, 3, 3))
  theta <- angles[1:Z]
  varphi <- angles[(Z+1):(2*Z)]
  eta <- angles[(2*Z+1):(3*Z)]
  gamma1 <- cbind(cos(varphi), sin(varphi)*cos(eta), sin(varphi)*sin(eta))
  gamma2 <- cbind(
    -cos(theta)*sin(varphi), 
    cos(theta)*cos(varphi)*cos(eta)-sin(theta)*sin(eta), 
    cos(theta)*cos(varphi)*sin(eta)+sin(theta)*cos(eta))
  gamma3 <- cbind(
    sin(theta)*sin(varphi), 
    -sin(theta)*cos(varphi)*cos(eta)-cos(theta)*sin(eta),
    -sin(theta)*cos(varphi)*sin(eta)+cos(theta)*cos(eta))
  G <- abind(gamma1, gamma2, gamma3, along=3)
  cluster_norms <- sqrt(apply(G^2, 1, colSums))
  for(z in 1:Z){
    G[z,,] <- G[z,,] %*% diag(1.0/cluster_norms[, z])
  }
  return(G)
}

correct_cos <- function(raw_values){
  raw_values[raw_values>1 & raw_values<1+EPSILON] = 1
  raw_values[raw_values< -1 & raw_values> -1-EPSILON] = -1
  return(raw_values)
}

G2angles <- function(G){
  varphi <- acos(correct_cos(G[, 1, 1]))
  theta <- acos(correct_cos(-G[,1,2]/sqrt(1-correct_cos(G[,1,1])^2)))
  eta <- acos(correct_cos(G[,2,1]/sqrt(1-correct_cos(G[,1,1])^2)))
  eta[G[,3,1]<0] <- 2*pi - eta[G[,3,1]<0]
  indice <- sqrt(1-correct_cos(G[,1,1])^2) == 0
  eta[indice] <- pi
  theta[indice] <- acos(correct_cos(-G[indice, 3, 3]))
  return(c(theta, varphi, eta))
}

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



# Visualization of confusion matrix

visual.table <- data.table(
  ground_truth = as.factor(rep(1:3, 3*3)),
  estimated = as.factor(rep(rep(1:3, each=3), 3)),
  count = c( table(id.true, id.dir1), table(id.true, id.km1), table(id.true, id.vonMF1)),
  method = as.factor(rep(c('Mixture kent', 'K-mean',"Mixture Vonmises"), each=3*3))
)

ggplot(data = visual.table, aes(x=estimated, y=ground_truth, fill=count)) + 
  geom_tile() +
  geom_text(aes(label=count), color='white') +
  scale_y_discrete(limits = rev(levels(visual.table$ground_truth))) +
  coord_equal() +
  facet_wrap(~method)



# visualization of spherical of true cluster
df <- data.table(x)
colnames(df) <- c('x', 'y', 'z')
df$true_cluster <- as.character(id.true)
df.arrow <- data.table(rbind(matrix(0, Z, 3), true.G))
colnames(df.arrow) <- c('x', 'y', 'z')
df.arrow[, true_cluster:=rep(c(1, 2, 3), 2)]
df.arrow



library(pracma)
n_lat <- 16
n_lon <- 32
resolution <- 100
theta <- seq(-pi/2, pi/2, length.out=n_lat)
phi <- seq(0, 2*pi, length.out=resolution)
mgrd <- meshgrid(phi, theta)
phi <- mgrd$X
theta <-  mgrd$Y
radius <- 0.99
x <- cos(theta) * cos(phi) * radius
dim(x) <- NULL
y <- cos(theta) * sin(phi) * radius
dim(y) <- NULL
z <- sin(theta) * radius
dim(z) <- NULL
df.lat <- data.table(x, y, z)
df.lat[, group:=rep(1:n_lat, resolution)]
theta <- seq(-pi/2, pi/2, length.out=resolution)
phi <- seq(0, 2*pi, length.out=n_lon)
mgrd <- meshgrid(phi, theta)
phi <- mgrd$X
theta <-  mgrd$Y
radius <- 0.99
xx <- cos(theta) * cos(phi) * radius
dim(xx) <- NULL
yy <- cos(theta) * sin(phi) * radius
dim(yy) <- NULL
zz <- sin(theta) * radius
dim(z) <- NULL
df.lon <- data.table(xx, yy, zz)
df.lon[, group:=rep(1:n_lon, each=resolution)]


library(plotly)

p <- plot_ly(x=~x, y=~y, z=~z) %>%
  add_trace(data=df, type='scatter3d', mode='markers', size=1, color=~true_cluster) %>% 
  add_trace(data=df.arrow, type='scatter3d', mode='lines', split=~true_cluster, color=~true_cluster, line = list(width = 5)) %>%
  add_trace(data=df.lat, type='scatter3d', mode='lines', split=~group, line = list(width = 2, color = NULL, dash='dot')) %>%
  add_trace(data=df.lon, type='scatter3d', mode='lines', split=~group, line = list(width = 2, color = NULL, dash='dot'))
p






















