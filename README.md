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
