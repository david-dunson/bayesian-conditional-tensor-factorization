# [Bayesian Conditional Tensor Factorization](https://doi.org/10.1080/01621459.2015.1029129)
Yun Yang and David B. Dunson

## Code
This code is the two stage algorithm for tensor factorization model. Simulation is the main algorithm.
N is the total sample size; n is the training size; p is the number of features; d is the number of levels for each feature; x: N by p, is the features; y: N by 1, is the response.

logml.m is used when the response has two levels {0,1}  (response should be transformed into 0 or 1)
logml2.m is used when the response has three levels {1,2,3}  (response should be transformed into 1, 2 or 3)


## Abstract
In many application areas, data are collected on a categorical response and high-dimensional categorical predictors, with the goals being to build a parsimonious model for classification while doing inferences on the important predictors. In settings such as genomics, there can be complex interactions among the predictors. By using a carefully structured Tucker factorization, we define a model that can characterize any conditional probability, while facilitating variable selection and modeling of higher-order interactions. Following a Bayesian approach, we propose a Markov chain Monte Carlo algorithm for posterior computation accommodating uncertainty in the predictors to be included. Under near low-rank assumptions, the posterior distribution for the conditional probability is shown to achieve close to the parametric rate of contraction even in ultra high-dimensional settings. The methods are illustrated using simulation examples and biomedical applications. Supplementary materials for this article are available online.
