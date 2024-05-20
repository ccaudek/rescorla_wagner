data {
  int NS;  // number of subjects
  int MT;  // maximum number of trials
  int NC;  // number of choices (2)
  array[NS] int NT;  // number of trials per subject
  int K;  // number of coefficients for glm
  array[NS, MT] real<lower=-1,upper=1> rew;  // subject x trial reward: -1 punishment; 1 reward; -1 for missed
  array[NS, MT] int choice;  // chosen option: 1 chose left; 0 chose right; -1 for missed
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
  real<lower=0> a1;
  real<lower=0> a2;
  vector[K] b_mean;
  vector<lower=0>[K] b_sd;
  cholesky_factor_corr[K] Lcorr;

  array[NS] real<lower=0,upper=1> alpha_raw;
  matrix[K, NS] beta_raw;
}

transformed parameters {
  array[NS] real<lower=0,upper=1> alpha;
  matrix[NS, K] beta;

  for (s in 1:NS) {
    alpha[s] = Phi_approx(a1 * alpha_raw[s]);
    beta[s] = (b_mean + diag_pre_multiply(b_sd, Lcorr) * beta_raw[,s])';
  }

  array[NS, MT, NC] real Q;
  array[NS, MT] real delta;

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
      if (t == 1) {
        for (c in 1:NC) {
          Q[s,t,c] = 0.5;
        }
        delta[s,t] = 0;
      }
      if (choice[s,t] >= 1 && choice[s,t] <= NC) {
        delta[s,t] = rew[s,t] - Q[s,t,choice[s,t]];
        if (t < NT[s]) {
          Q[s,t+1,choice[s,t]] = Q[s,t,choice[s,t]] + alpha[s] * delta[s,t];
          Q[s,t+1,3-choice[s,t]] = Q[s,t,3-choice[s,t]];
        }
      } else {
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
  a1 ~ cauchy(0, 2.5);
  a2 ~ cauchy(0, 2.5);
  b_mean ~ normal(0, 1);
  b_sd ~ cauchy(0, 2.5);
  alpha_raw ~ beta(2, 2);
  to_vector(beta_raw) ~ normal(0, 1);

  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        choice_two[s,t] ~ bernoulli_logit(beta[s,1] + beta[s,2] * (Q[s,t,1] - Q[s,t,2]));
      }
    }
  }
}

generated quantities {
  matrix[K, K] Omega;
  matrix[K, K] Sigma;
  array[N] real log_lik;
  int n;

  Omega = multiply_lower_tri_self_transpose(Lcorr);
  Sigma = quad_form_diag(Omega, b_sd);

  n = 1;
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        log_lik[n] = bernoulli_logit_lpmf(choice_two[s,t] | beta[s,1] + beta[s,2] * (Q[s,t,1] - Q[s,t,2]));
        n = n + 1;
      }
    }
  }
}
