// Code for Bayesian regression w/ bivariate response and nested groups
// see https://mc-stan.org/docs/2_21/stan-users-guide/hierarchical-logistic-regression.html
// and https://mc-stan.org/docs/2_21/stan-users-guide/multivariate-outcomes.html
// for details
data {
  int<lower=0> N; // Number of outcomes
  int<lower=1> P; // Number of predictors
  int<lower=1> J; // Number of group 1 (neighbourhood, nested in group 2)
  int<lower=1> K; // Number of group 2 (borough)
  int<lower=1,upper=J> group1;
  int<lower=1,upper=K> group2;
  vector[2] y[N]; // Response (bivariate, each containing N samples)
  vector[P] x[N]; // Predictors (P-variate, each containing N observations)
}
parameters {
  matrix[2,P] beta; // 2 x P matrix of coefficients
  cholesky_factor_corr[2] L_Omega; // Cholesky factor of correlation matrix, Omega
  vector<lower=0>[2] L_sigma; // (Scaled) vector of 2 standard deviation parameters
}
model {
  // Stan does not implement Gibbs, so covariance prior is here not Wishart
  // Cholesky parametrization is for efficiency
  // LKJ prior is recommended by the Stan authors
  vector[2] mu[N]; // Linear predictor (normal mean)
  matrix[2,2] L_Sigma; // Actual covariance matrix (Chol.-parametrized)
  
  for (n in 1:N)
    mu[n] = beta * x[n];
  L_Sigma = diag_pre_multiply(L_sigma, L_Omega);
  
  to_vector(beta) ~ normal(0, 1); // Must be scaled for this to work
  L_Omega ~ lkj_corr_cholesky(0.5); // 0 < shape parameter <1 favors strong correlation
  L_sigma ~ cauchy(0, 1); // Same; half-Cauchy
  
  y ~ multi_normal_cholesky(mu, L_Sigma);
}
