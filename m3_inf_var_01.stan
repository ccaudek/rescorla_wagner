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
  matrix<lower=0, upper=1>[NS, 2] alpha; // subject-level alpha
  matrix[NS, 3] beta; // subject-level beta
}

transformed parameters {
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
  // Priors for alpha
  alpha[:,1] ~ beta(0.5, 0.5);
  alpha[:,2] ~ beta(0.5, 0.5);

  // Priors for beta
  beta[:,1] ~ normal(0, 5);
  beta[:,2] ~ normal(0, 5);
  beta[:,3] ~ normal(0, 5);

  // Likelihood
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        choice_two[s,t] ~ bernoulli_logit(beta[s,1] +
                                          beta[s,2] * (Q[s,t,1] - Q[s,t,2]) +
                                          beta[s,3] * stickiness[s,t]);
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
                                          beta[s,1] +
                                          beta[s,2] * (Q[s,t,1] - Q[s,t,2]) +
                                          beta[s,3] * stickiness[s,t]);
        n = n + 1;
      }
    }
  }
}
