

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




















