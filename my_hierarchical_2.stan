data {
  int NS;  // number of subjects
  int MT;  // maximum number of trials
  int NC;  // number of choices (2)
  array[NS] int NT;  // number of trials per subject
  int K;  // number of coefficients for glm
  array[NS, MT] real<lower=-1,upper=1> rew;  // subject x trial reward, -1 for missed
  array[NS, MT] int choice;  // chosen option, -1 for missed
  array[NS, MT] int choice_two;  // 1=chose left, 0=chose right, -1 for missed
}

transformed data {
  int N = 0;
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        N = N + 1;
      }
    }
  }
}

parameters {
  // parameters for group alpha distribution
  real<lower=0> a1;
  real<lower=0> a2;

  // parameters for group beta distributions
  vector[K] b_mean;
  vector<lower=0>[K] b_sd;

  // subject-level alpha and betas
  array[NS] real<lower=0,upper=1> alpha;
  matrix[NS,K] beta;
  
  // cholesky factorization of correlation matrix of subject-level estimates
  cholesky_factor_corr[K] Lcorr;
}

transformed parameters {
  // subject x trials x choice Q value matrix
  array[NS, MT, NC] real Q;

  // prediction error matrix
  array[NS, MT] real delta;

  // initialize Q and delta to avoid NaNs
  for (s in 1:NS) {
    for (m in 1:MT) {
      for (c in 1:NC) {
        Q[s,m,c] = 0.0;
      }
    }
  }

  delta = rep_array(0.0, NS, MT);

  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      // set initial values of Q and delta on first trial
      if (t == 1) {
        for (c in 1:NC) {
          Q[s,t,c] = 0.5;  // initialize Q values
        }
        delta[s,t] = 0;
      }
      if (choice[s,t] >= 1 && choice[s,t] <= NC) {  // ensure choice is within valid range
        // PE = reward - expected
        delta[s,t] = rew[s,t] - Q[s,t,choice[s,t]];

        if (t < NT[s]) {
          // update value with alpha-weighted PE
          Q[s,t+1,choice[s,t]] = Q[s,t,choice[s,t]] + alpha[s] * delta[s,t];
          // value of unchosen option is not updated
          Q[s,t+1,3-choice[s,t]] = Q[s,t,3-choice[s,t]];  // use 3-choice to get the other option
        }
      } else {
        // if no valid response, keep Q value and set delta to 0
        if (t < NT[s]) {
          for (c in 1:NC) {
            Q[s,t+1,c] = Q[s,t,c];
          }
          delta[s,t] = 0;
        }
      }
    }
  }
}



model {
  // hyperpriors
  a1 ~ cauchy(0,5);
  a2 ~ cauchy(0,5);

  b_mean ~ normal(0,5);
  b_sd ~ cauchy(0,5);

  // distributions of subject effects
  alpha ~ beta(0.5,0.5);
  beta[:,1] ~ normal(0,5);
  beta[:,2] ~ normal(0,5);
  
  for (s in 1:NS) {
    beta[s] ~ multi_normal_cholesky(b_mean, diag_pre_multiply(b_sd, Lcorr));
  }
  
  // data generating process (likelihood)
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        if (t < 76) {
          choice_two[s,t] ~ bernoulli_logit(beta[s,1] + beta[s,2] * (Q[s,t,1] - Q[s,t,2]));
        }
      }
    }
  }
}

generated quantities {
  matrix[K,K] Omega;
  matrix[K,K] Sigma;
  array[N - (30 * NS)] real log_lik_old;
  array[30 * NS] real log_lik;
  int n;
  int m;

  // get correlation matrix from cholesky
  Omega = multiply_lower_tri_self_transpose(Lcorr);

  // diag_matrix(b_sd) * Omega * diag_matrix(b_sd) to get covariance
  Sigma = quad_form_diag(Omega, b_sd);

  n = 1;
  m = 1;
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        if (t <= 75) {
          log_lik_old[n] = bernoulli_logit_lpmf(choice_two[s,t] | beta[s,1] + beta[s,2] * (Q[s,t,1] - Q[s,t,2]));
          n = n + 1;
        }
        if (t > 75) {
          log_lik[m] = bernoulli_logit_lpmf(choice_two[s,t] | beta[s,1] + beta[s,2] * (Q[s,t,1] - Q[s,t,2]));
          m = m + 1;
        }
      }
    }
  }
}
