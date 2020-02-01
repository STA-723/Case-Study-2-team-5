// Code for Bayesian regression w/ bivariate response and nested groups
// see https://mc-stan.org/docs/2_21/stan-users-guide/hierarchical-logistic-regression.html
// and https://mc-stan.org/docs/2_21/stan-users-guide/multivariate-outcomes.html
// for details
data {
  int<lower=0> N; // Number of outcomes
  int<lower=1> P; // Number of predictors
  int<lower=1> J; // Number of group 1 (neighbourhood, nested in group 2)
  int<lower=1> K; // Number of group 2 (borough)
  int<lower=1,upper=J> group1[N];
  int<lower=1,upper=K> group2[N];
  vector[2] y[N]; // Response (bivariate, each containing N samples)
  vector[P] x[N]; // Predictors (P-variate, each containing N observations)
}
parameters {
  matrix[2,P] beta_g1[J];
  matrix[2,P] beta_g2[K];
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=-1,upper=0> rho;
  real<lower=0> tau1;
  real<lower=0> tau2;
}
model {
  vector[2] mu[N]; // Linear predictor (normal mean)
  matrix[2,2] Sigma; // Actual covariance matrix (Chol.-parametrized)
  for (n in 1:N) {
    mu[n] = beta_g1[group1[n]] * x[n];
    to_vector(beta_g1[group1[n]][1]) ~ normal(to_vector(beta_g2[group2[n]][1]), tau1);
    to_vector(beta_g1[group1[n]][2]) ~ normal(to_vector(beta_g2[group2[n]][2]), tau2);
    to_vector(beta_g2[group2[n]][1]) ~ normal(0, 1);
    to_vector(beta_g2[group2[n]][2]) ~ normal(0, 1);
  }
  sigma1 ~ normal(0, 1);
  sigma2 ~ normal(0, 1);
  rho ~ uniform(-1, 0);
  tau1 ~ normal(0, 1);
  tau2 ~ normal(0, 1);
  Sigma[1,1] = sigma1^2;
  Sigma[1,2] = rho * sigma1 * sigma2;
  Sigma[2,1] = Sigma[1,2];
  Sigma[2,2] = sigma2^2;
  
  y ~ multi_normal(mu, Sigma);
}
