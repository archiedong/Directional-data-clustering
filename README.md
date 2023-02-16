# Directional-data-clustering
To cluster asymmetrically distributed data on a sphere, Kent mixture model is commonly used. However, the performanc of such a model can be severely affected by the presence of noise or outliers. A novel contaminated Kent mixture model is proposed to alleviate this issue effectively.

# Introduction
![img|2724x1365, 50%](https://user-images.githubusercontent.com/60518209/219405639-6b425462-90af-41fd-9606-663ab99ac721.png)
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
\ddot{\tau}_{ik}= \frac{\dot\pi_k \mathcal{K}(x_i; \dot\vartheta_k)} {\sum \dot\pi_r \mathcal{K}(x_i; \dot\vartheta_r)}
```
where one dot and two dots on the top of parameters stand for estimates at the previous and current iterations, respectively. Here, $\tau_{ik}$ denotes the probability that $x_i$ originates from the $k^{th}$ mixture component. The conditional expectation of the complete-data log-likelihood commonly referred to as the $Q$ function, takes the following form:
```math
  \begin{split}
    Q(\Theta; \dot\Theta, \{x_i\}_{i=1}^n) =  \sum_{i=1}^{n} \sum_{k=1}^{K} \ddot\tau_{ik}\bigg(&\log \pi_k - \log\bigg\{2\pi \sum_{j=0}^\infty \frac{\Gamma(j+\frac{1}{2})}{\Gamma(j+1)}\beta_k^{2j} \left(\frac{2}{\kappa_k}\right)^{2j+\frac{1}{2}} I_{2j+\frac{1}{2}}(\kappa_k)\bigg\}\\
 & + \kappa_k \gamma_{1k}^\top x_i + \beta_k \gamma_{2k}^\top x_i x_i^\top \gamma_{2k} - \beta_k \gamma_{3k}^\top x_i x_i^\top \gamma_{3k}\bigg).
    \end{split}
```
The maximization of this $Q$ function requires numerical optimization over all model parameters except for the mixing proportions that can be estimated analytically by $$\ddot\pi_k = n^{-1} \sum_{i = 1}^n \ddot\tau_{ik}.$$.

## Contaminated Kent mixture model
Despite its attractive capability of modeling ellipticity on a sphere, the performance of KMM can be severely affected by the presence of noise or outliers. In this section, we propose a novel distribution named a contaminated Kent mixture model that can relax these limitations effectively. Similarly to the mixture with contaminated Gaussian components, the underlying assumption is that the excessive variability in each data group can be modeled by a submixture of two components sharing the majority of their parameters but having different dispersions. In a Kent distribution, the concentration parameter $\kappa$ reflects the variability in data. Hence, the pdf of CKMM is given by
```math
g(x;\Theta) = \sum_{k=1}^{K} \pi_k \big[\delta_k \mathcal{K}(x; \vartheta_k) + (1 - \delta_{k})\mathcal{K}(x; \tilde\vartheta_k)\big],
```
where $\vartheta_k = (\kappa_k, \beta_k, \theta_k, \varphi_k, \eta_k)^\top$, $\tilde\vartheta_k = (\alpha_k \kappa_k, \beta_k, \theta_k, \varphi_k, \eta_k)^\top$, and $\delta_k$ is a submixture weight parameter. Here, the only difference between $\vartheta_k$ and $\tilde\vartheta_k$ vectors is due to the multiplier $\alpha_k \in (0, 1)$ responsible for the dispersion inflation in the second subcomponent. The purpose of introducing the inflated subcomponent is to better model heavy tails. From now on, we refer to the uninflated and inflated subcomponents as primary and secondary, respectively.
The corresponding complete-data likelihood function can be written as
```math
 L_c(\Theta; \{x_i\}_{i=1}^n) = \prod_{i=1}^n \prod_{k=1}^K \bigg[\pi_k \Big[\delta_{kj} \mathcal{K}(x_i; \vartheta_k)\Big]^{I(W_i = 1|Z_i = k)}\Big[(1 - \delta_{k})\mathcal{K}(x_i; \tilde\vartheta_k)\Big]^{I(W_i = 2|Z_i = k)}\bigg]^{I(Z_i = k)},
```
where $W_i$ represents the subcomponent label, with $W_i = 1$ and $W_i = 2$ implying that the observation $x_i$ originated from the primary and secondary subcomponents, respectively. Then, it follows that the E-step of the EM algorithm requires updating posterior probabilities according to the following expressions:
```math
  \ddot{\tau}_{ik} = \frac{\dot{\pi_k} \big[\dot{\delta}_k \mathcal{K}(x_i; \dot\vartheta_k) + (1 - \dot{\delta}_{k})\mathcal{K}(x_i; \dot{\tilde\vartheta}_k)\big]}{\sum_{r = 1} ^{K}\dot{\pi_r} \big[\dot{\delta}_r \mathcal{K}(x_i; \dot\vartheta_r) + (1 - \dot{\delta}_{r})\mathcal{K}(x_i; \dot{\tilde\vartheta}_r)\big]}, \quad \ddot{\nu}_{i|k} = \frac{\dot{\delta}_k \mathcal{K}(x_i; \dot\vartheta_k) }{\dot{\delta}_k \mathcal{K}(x_i; \dot\vartheta_k) + (1 - \dot{\delta}_{k})\mathcal{K}(x_i; \dot{\tilde\vartheta}_k)}.
```
Here, $\tau_{ik}$ is the probability that $x_i$ originates from the $k^{th}$ mixture component and $\nu_{i|k}$ is the probability that $x_i$ belongs to the primary distribution within the $k^{th}$ component. In other words, $\nu_{i|k}$ can be seen as the probability that $x_i$ is not an outlying observation in the $k^{th}$ component.
The $Q$ function that needs to be optimized at the M step takes the following form:
```math
  \begin{split}
Q(\Theta; \dot{\Theta}, \{x_i\}_{i=1}^n) =& \sum_{i=1}^{n} \sum_{k=1}^{K} \ddot{{\tau}}_{ik}\Big[\log{\pi_k} + \ddot{{\nu}}_{i|k} \log{\delta_k} + (1 - \ddot{\nu}_{i|k}) \log{(1 -\delta_k)} \\
& + \ddot{\nu}_{i|k} \log{\mathcal{K} (x_i; \vartheta_k)} + (1 - \ddot{\nu}_{i|k})\log{\mathcal{K}(x_i; \tilde\vartheta_k)}\Big].
  \end{split}
```
Taking partial derivatives of Equation with respect to $\pi_k$ and $\delta_k$ leads to the following expressions:
```math
\ddot{\pi}_{k} = \frac{\sum \ddot{\tau}_{ik}}{n}, \quad \quad \ddot{\delta}_{k} = \frac{\sum \ddot{\tau}_{ik} \ddot{\nu}_{i|k}}{\sum \ddot{\tau}_{ik}}.
```
The performance of the EM algorithm depends on a chosen initialization strategy. In this paper, vMFMM is initialized by the partition obtained by a directional $k$-means algorithm implemented in the R package skmeans. Unfortunately, the use of the clustering solution obtained by skmeans for initializing KMM as well as CKMM does not show satisfactory results. Therefore, to start KMM, we use the partition obtained from vMFMM, and to initialize CKMM, we employ the clustering found by KMM. Such a strategy produces the best results for the three considered mixture models.

# Simulation
<p align="center">
  <img src="https://user-images.githubusercontent.com/60518209/219496117-7e3d5f8e-f261-4f0a-9fcf-28d21c75c17e.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219496149-c277c65c-4c70-447a-9028-ffd40d32a775.png" width="230" /> 
  <img src="https://user-images.githubusercontent.com/60518209/219496174-3f17c29b-3bfe-440c-9ec0-9ff2474307f7.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219496235-f256e937-3045-4d15-9f14-daadab94779a.png" width="230" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/60518209/219497470-b4aeffc1-5499-4968-9d08-967c47e05f07.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219497486-62dd64c9-d08f-4eee-93ef-0ef4550b02f4.png" width="230" /> 
  <img src="https://user-images.githubusercontent.com/60518209/219497490-5b98389e-f2af-4e33-85a6-ec9f41e92a10.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219497493-78ad31b4-d8fa-462f-9707-b01a13112c64.png" width="230" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/60518209/219498070-d3724dd3-9eb3-46aa-9717-092b9c62ffee.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219498081-a2c7c76d-6cac-4aad-9570-10206d34ca31.png" width="230" /> 
  <img src="https://user-images.githubusercontent.com/60518209/219498085-4a446b9e-e30e-4679-884e-444db155af49.png" width="230" />
  <img src="https://user-images.githubusercontent.com/60518209/219498091-e76bd02d-59b1-48ae-a075-9785fc060963.png" width="230" />
</p>


In this section, we assess the proposed methodology by comparing the performance of vMFMM, KMM, and CKMM in three different settings. In the first setting, we simulate data from vMFMM. In the second one, data are generated from KMM. Finally, in the last case data sets are obtained based on CKMM. In each setting, there are 200 data sets of size 2500 generated from mixtures with parameters provided in Table~\ref{table.exp}. All mixtures are assumed to have five components. For each simulated data set, we apply vMFMM, KMM, and CKMM. To assess the adequacy of model fitting, we use BIC. As the true number of clusters is assumed to be unknown, the best model is identified by examining BIC values for $K = 1, \ldots, 20$. In order to evaluate model-based clustering performance, we compare the original simulated and estimated partitions based on the adjusted Rand index (ARI) using function RandIndex from the R package MixSim. The upper bound of ARI is equal to 1.0 and is achievable only in the case of the perfect match between the compared partitions. The obtained results are summarized in Table1 and illustrated in Figure2.

In Table1, one can observe the distribution of the estimated mixture order $\hat K$. The number of correct model detections is highlighted with the bold font. We also report the average maximized log-likelihood value, BIC, and ARI corresponding to the best identified models. As we can see, when the data are simulated from vMFMM, all three competitors perform similarly in terms of model fitting, with corresponding log-likelihood values being just marginally different. As expected, vMFMM produces the smallest BIC value as the number of model parameters is the lowest among the three models. All mixtures identify the correct number of components and demonstrate identical clustering performance as reflected by the average ARI value (0.868). In the setting when the data are generated from KMM, we observe that KMM shows the best results in terms of BIC and the performance of vMFMM degrades considerably. In particular, its average log-likelihood value is 55 units  lower than that of KMM and CKMM. The reason for that is the incapability of vMFMM to model the ellipticity imposed by KMM. As a result, more components (9 to 14) are needed to fit the data adequately. This affects the clustering results as reflected by relatively low ARI (0.466). KMM and CKMM perform considerably better, being able to identify the correct number of components in all cases and producing average ARI values equal to 0.805 and 0.806, respectively. In the final setting, the data are simulated from CKMM. The best performer is CKMM, with the correct mixture order identified in all cases and average log-likelihood value equal to 116.8. The corresponding BIC is 71.5, the lowest among the competitors. Both alternative approaches, vMFMM and KMM, exhibit mixture order overestimation that results in unsatisfactory clustering performance, with vMFMM being more severely affected as reflected by the number of estimated components ranging from 8 to 16. The presence of scatter results in the detection of 4 to 11 components by KMM. It is interesting to notice that there are 65 cases of the mixture order underestimation by KMM. This is a result of the substantial overlap between components which in the presence of scatter, leads to merging some of them. Another interesting observation is that the ARI value associated with KMM is lower than that of vMFMM. This can be explained by a tendency of ARI to penalize underestimation more severely than overestimation. The discussion of the performance of vMFMM, KMM, and CKMM is supported by illustrations provided in Figure2. Each column of plots in the figure represents one of the three settings considered. The first row of plots illustrates a sample data set simulated in each case, while second, third, and forth rows present clustering solutions obtained by vMFMM, KMM, and CKMM, respectively.

To conclude, the clustering performance of CKMM is substantially better than that of competitiors as can be seen from much higher average ARI. Based on the conducted set of experiments, we can conclude that while the correct mixture model is always prefered by BIC, the model-based clustering performance of CKMM is superior over its competitors as demonstrated by the best results obtained in all considered settings.

<img width="8500" align = "center" alt="Screenshot 2023-02-16 at 4 18 23 PM" src="https://user-images.githubusercontent.com/60518209/219500073-9948add7-479e-431a-bb81-d6186b727687.png">

<img width="8500" align = "center" alt= "Screenshot 2023-02-16 at 4 18 47 PM" src="https://user-images.githubusercontent.com/60518209/219500421-d247a100-9e57-4636-97ba-fab3fff15d0c.png">


# Applications
<img width="8500" align = "center" alt= "https://user-images.githubusercontent.com/60518209/219501607-b29c2b2e-f258-49ef-b583-2a4177323f7f.png">
<img width="8500" align = "center" alt= "https://user-images.githubusercontent.com/60518209/219501782-aa597309-fd73-429b-b48e-63f045cd7b43.png">

To demonstrate the importance of the methodology outlined in this paper, we consider a data set referred to as {\it Rock Site} data (generously provided by the Julius Kruttschnitt Mineral Reseach Centre (JKMRC) at the University of Queensland). In nature, a rock usually contains multiple fractures or discontinuity planes that are generally formed in a parallel fashion. Such parallel fracture planes are commonly called joint sets. Due to the fact that forces from more than one direction often act on a rock mass, a number of joint sets may be observed. In order to record the orientation of the joint sets in space, the dip direction and dip angle are measured. As shown in Figure4, the dip direction is the direction toward which the plane is inclined and the dip angle is the angle of  the plane inclination. The {\it Rock Site} data set consists of 860 measurements of dip angle and direction.  %In this application, the data are axial (antipodally symmetric) and the vMFMM, KMM, and CKMM were fitted to the data. %One advantage of above three methods is that their use is not restricted to antipodally symmetric data.

We apply vMFMM, KMM, and CKMM to the data. The data and specific model fit can be observed in Figure5. As we can see, there is an abundance of observations representing scatter or associated with heavy tails of data groups. Figure5 illustrates the performance of the three considered mixture models in terms of BIC. As we can see, the best performer is CKMM. Its BIC is the lowest for all mixture orders, with the optimal number of components detected for $K = 5$. The corresponding BIC is 2402. The second best performer is KMM that finds a solution with seven mixture components and BIC equal to 2471. Figure4 shows that the extra two components are associated with heavy tails of magenta and orange components from the CKMM solution. The worst performance is shown by vMFMM. It prefers a 10-component solution with roughly spherical clusters. The BIC value associated with this solution is 2612. The considered application confirms that the proposed methodology is useful in modeling data with heavy tails or scatter.

