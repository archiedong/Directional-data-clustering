


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





















