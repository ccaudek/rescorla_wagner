data {
  int NS; // number of subjects
  int MT; // maximum number of trials
  int NC; // number of choices (2)
  array[NS] int NT; // number of trials per subject
  array[NS, MT] real<lower=-1, upper=1> rew; // subject x trial reward
  array[NS, MT] int choice; // chosen option
  array[NS, MT] int choice_two; // 1 = chose left, 0 = chose right
  array[NS, MT] int stickiness; // 1 = previous choice left, -1 = previous choice right
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
  matrix[NS, 2] alpha_logit_raw; // subject-level alpha on logit scale
  matrix[NS, 3] beta_raw; // subject-level beta
  vector<lower=0>[NS] inv_temp_raw; // inverse temperature parameter

  // Hyperparameters
  vector[2] mu_alpha_logit;
  vector<lower=0>[2] sigma_alpha_logit;
  vector[3] mu_beta;
  vector<lower=0>[3] sigma_beta;
  real<lower=0> mu_inv_temp;
  real<lower=0> sigma_inv_temp;
}

transformed parameters {
  matrix<lower=0, upper=1>[NS, 2] alpha;
  matrix[NS, 2] alpha_logit;
  matrix[NS, 3] beta;
  vector<lower=0>[NS] inv_temp;

  for (s in 1:NS) {
    for (i in 1:2) {
      alpha_logit[s,i] = mu_alpha_logit[i] + sigma_alpha_logit[i] * alpha_logit_raw[s,i];
      alpha[s,i] = inv_logit(alpha_logit[s,i]);
    }

    for (i in 1:3) {
      beta[s,i] = mu_beta[i] + sigma_beta[i] * beta_raw[s,i];
    }

    inv_temp[s] = mu_inv_temp + sigma_inv_temp * inv_temp_raw[s];
  }

  array[NS, MT, NC] real Q; // subject x trials x choice Q value matrix
  array[NS, MT] real delta; // prediction error matrix

  // Initialize Q and delta
  for (s in 1:NS) {
    for (t in 1:MT) {
      for (c in 1:NC) {
        Q[s,t,c] = 0.5; // initial Q values
      }
      delta[s,t] = 0.0;
    }
  }

  // Update Q and delta
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (t == 1) {
        delta[s,t] = 0;
      } else if (choice[s,t] > 0) {
        // PE = reward - expected
        delta[s,t] = rew[s,t] - Q[s,t,choice[s,t]];

        // Update value with alpha-weighted PE
        if (t < NT[s]) {
          if (delta[s,t] >= 0) {
            Q[s,t+1,choice[s,t]] = Q[s,t,choice[s,t]] + alpha[s,1] * delta[s,t];
          } else {
            Q[s,t+1,choice[s,t]] = Q[s,t,choice[s,t]] + alpha[s,2] * delta[s,t];
          }

          // Value of unchosen option is not updated
          Q[s,t+1,3-choice[s,t]] = Q[s,t,3-choice[s,t]]; // 3-choice[s,t] to toggle between 1 and 2
        }
      }
    }
  }
}

model {
  // Hyperpriors
  mu_alpha_logit ~ normal(0, 1);
  sigma_alpha_logit ~ cauchy(0, 2.5);
  mu_beta ~ normal(0, 1);
  sigma_beta ~ cauchy(0, 2.5);
  mu_inv_temp ~ normal(1, 0.5);
  sigma_inv_temp ~ cauchy(0, 2.5);

  // Priors for subject-level parameters
  to_vector(alpha_logit_raw) ~ normal(0, 1);
  to_vector(beta_raw) ~ normal(0, 1);
  inv_temp_raw ~ normal(0, 1);

  // Likelihood
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        choice_two[s,t] ~ bernoulli_logit(inv_temp[s] * (beta[s,1] +
                                          beta[s,2] * (Q[s,t,1] - Q[s,t,2]) +
                                          beta[s,3] * stickiness[s,t]));
      }
    }
  }
}

generated quantities {
  array[N] real log_lik;
  int n = 1;

  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        log_lik[n] = bernoulli_logit_lpmf(choice_two[s,t] |
                                          inv_temp[s] * (beta[s,1] +
                                          beta[s,2] * (Q[s,t,1] - Q[s,t,2]) +
                                          beta[s,3] * stickiness[s,t]));
        n = n + 1;
      }
    }
  }
}
