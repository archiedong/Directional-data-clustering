# Directional-data-clustering
To cluster asymmetrically distributed data on a sphere, Kent mixture model is commonly used. However, the performanc of such a model can be severely affected by the presence of noise or outliers. A novel contaminated Kent mixture model is proposed to alleviate this issue effectively.

# Introduction
Directional statistics is a popular area with an abundance of applications in earth science, meteorology, biology, and other disciplines
studying data with periodic characteristics. Directional data represent a collection of unit vectors with information contained in the
form of vector orientation. Since there is no natural zero direction, the analysis of directional data doesn’t fall in the framework of
traditional statistical methods relying on distributions in Euclidean space. Hence, alternative distributions for describing directional
data are employed, among which the von Mises-Fisher distribution (Mardia and Jupp 2009) is the most famous and broadly used. This
distribution along with a wrapped symmetric Gaussian distribution (Collett and Lewis 1981) are capable of modeling spherical data
lying on a unit hypersphere. However, modeling elliptically distributed directional data requires more sophisticated tools. In a threedimensional
space, Kent distribution (Kent 1982) is a popular alternative that provides higher modeling  exibility for elliptical data. For
modeling heterogeneous directional data, Kent mixture models (KMM) have been developed by Peel, Whiten, and McLachlan (2001a).
Unfortunately, the performance of such mixtures can be strongly a ected by the presence of noise or outliers. In this work, we propose
a novel contaminated Kent mixture model (CKMM) that can account for elliptical data on a sphere and automatically identify scatter
points e ectively.
The rest of the paper is organized as follows. In Section 2, the methodology related to KMM and CKMM is considered. Section 3
investigates the performance of CKMM in comparison with von Mises-Fisher mixture model (vMFMM) (Banerjee, Dhillon, Ghosh, and
Sra 2005) and KMM in a series of experiments. Section 4 is devoted to the application of the proposed methodology to the real-life
data. The paper concludes with a discussion provided in Section 5.

# Methodology
## Kent distribution
The Kent distribution (Kent 1982), also known as a  ve-parameter
Fisher-Bingham distribution, is analogous to a bivariate normal distribution
with an unrestricted covariance matrix in the sense that it
is capable of modeling elliptical data, although on a sphere. It is considerably
more  exible than a von Mises-Fisher distribution that is
similar to a normal distribution with a spherical covariance matrix.
The probability density function (pdf) of the Kent distribution is
given by 
```math
\mathcal{K}(\textbf{x}; \vartheta)= \frac{ \exp{(\kappa \gamma_1 ^\top x + \beta[(\gamma_2^\top x)^2 - (\gamma_3^\top x)^2])}}{2\pi \sum_{j=0}^\infty \frac{\Gamma(j+\frac{1}{2})}{\Gamma(j+1)}\beta^{2j} \left(\frac{2}{\kappa}\right)^{2j+\frac{1}{2}} I_{2j+\frac{1}{2}}(\kappa)}
```
![img|2724x1365, 50%](https://user-images.githubusercontent.com/60518209/219405639-6b425462-90af-41fd-9606-663ab99ac721.png)

where $I_\nu(\cdot)$ denotes the modified Bessel function of the first kind of order $\nu$ and $\vartheta = (\kappa, \beta, \theta, \varphi, \eta)$ is the parameter vector. Here, $\kappa$ stands for a concentration parameter and $\beta$ is responsible for modeling the ellipticity of distribution contours, subject to the inequality $0 < \beta /\kappa < 0.5$. $\gamma_1$, $\gamma_2$, $\gamma_3$ are orthogonal unit vectors representing mean, major, and minor directions, respectively, as shown in above Kent Figure. Finally, $\theta$, $\varphi$, and $\eta$ are the angles responsible for transforming $\gamma_1 = (\gamma_{11}, \gamma_{12}, \gamma_{13})^\top$, $\gamma_2= (\gamma_{21}, \gamma_{22}, \gamma_{23})^\top$, and $\gamma_3= (\gamma_{31}, \gamma_{32}, \gamma_{33})^\top$ into $x = (1, 0, 0)^\top$, $y = (0, 1, 0)^\top$, and $z = (0, 0, 1)^\top$ in Cartesian coordinates. 

The transformation matrix $R$ is such that $[x \; y\; z] = [\gamma_1 \; \gamma_2 \; \gamma_3] R^\top$ with its transpose defined as follows:
```math
\textbf{R}^T =
  \begin{bmatrix}
    \gamma_1^T\\
    \gamma_2^T\\
    \gamma_3^T 
  \end{bmatrix} \\
  =
  \begin{bmatrix}
    \cos{\varphi}& \sin{\varphi} \cos{\eta} & \sin{\varphi} \sin{\eta}\\
    -\cos\theta \sin\varphi & \cos\theta \cos\varphi \cos\eta - \sin\theta \sin\eta & \cos\theta \cos\varphi \sin\eta + \sin\theta \cos\eta\\
    \sin\theta \sin\varphi & -\sin\theta \cos\varphi \cos\eta - \cos\theta \sin\eta & -\sin\theta \cos\varphi \sin\eta + \cos\theta \cos\eta
  \end{bmatrix}.
```
Then, the relationship between the angles $\theta$, $\varphi$, $\eta$ and vectors $\gamma_1$, $\gamma_2$, $\gamma_3$ can be found as
```math
\begin{split}
\varphi = \arccos(\gamma_{11}),& \; \varphi \in{(0, \pi)},\quad\quad\quad\quad
\theta = \arccos\bigg(-\frac{\gamma_{21}}{\sqrt{1-\gamma_{11}}}\bigg), \; \theta \in{(0, \pi)},\\
&\eta = \arccos\bigg(\frac{\gamma_{12}}{\sqrt{1-\gamma_{11}}}\bigg), \; \eta \in{(0, 2\pi)}.
\end{split}
```

## Kent mixture model

Let $X_1, \ldots, X_n$ be a random sample consisting of $n$ observations lying on a sphere and following a mixture model with Kent-distributed components. Then, the corresponding pdf is given by
```math
g(x; \Theta) = \sum_{k=1}^{K} \pi_k \mathcal{K}(x;\vartheta_k),
```
where $\mathcal{K}(x; \vartheta_k)$ is the $k^{th}$ Kent mixture component pdf with parameter vector $\vartheta_k$ as provided in Equation 2. It can be shown that the E-step of the EM algorithm requires updating posterior probabilities according to the following expression:
```math
\ddot{\tau}_{ik}= \frac{\dot\pi_k \mathcal{K}(x_i; \dot\vartheta_k)}{ \sum_{r = 1}^{K} \dot\pi_r \mathcal{K}(x_i; \dot\vartheta_r)}
```
where one dot and two dots on the top of parameters stand for estimates at the previous and current iterations, respectively. Here, $\tau_{ik}$ denotes the probability that $x_i$ originates from the $k^{th}$ mixture component. The conditional expectation of the complete-data log-likelihood commonly referred to as the $Q$ function, takes the following form:
```math
  \begin{split}
    Q(\Theta; \dot\Theta, \{x_i\}_{i=1}^n) =  \sum_{i=1}^{n} \sum_{k=1}^{K} \ddot\tau_{ik}\bigg(&\log \pi_k - \log\bigg\{2\pi \sum_{j=0}^\infty \frac{\Gamma(j+\frac{1}{2})}{\Gamma(j+1)}\beta_k^{2j} \left(\frac{2}{\kappa_k}\right)^{2j+\frac{1}{2}} I_{2j+\frac{1}{2}}(\kappa_k)\bigg\}\\
 & + \kappa_k \gamma_{1k}^\top x_i + \beta_k \gamma_{2k}^\top x_i x_i^\top \gamma_{2k} - \beta_k \gamma_{3k}^\top x_i x_i^\top \gamma_{3k}\bigg).
    \end{split}
```
The maximization of this $Q$ function requires numerical optimization over all model parameters except for the mixing proportions that can be estimated analytically by $$\ddot\pi_k = n^{-1} \sum_{i = 1}^n \ddot\tau_{ik}$$.
