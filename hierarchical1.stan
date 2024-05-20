data {
  int<lower=1> NS; // Number of subjects
  int<lower=1> MT; // Maximum number of trials per subject
  array[NS] int NT; // Array containing the number of trials for each subject
  array[NS, MT] real rew; // Reward feedback: 1 for positive, -1 for negative, 0 for missed
  array[NS, MT] int choice; // Choices made by subjects, -1 for missed
}

parameters {
  real<lower=0> a1; // Parameter for beta distribution of alpha
  real<lower=0> a2; // Parameter for beta distribution of alpha
  array[NS] real alpha; // Subject-level learning rates
  real beta; // Coefficient for logistic regression
}

transformed parameters {
  array[NS, MT] real Q; // Q values for the chosen option
  array[NS, MT] real delta; // Prediction error

  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      // Initialize Q and delta for the first trial
      if (t == 1) {
        Q[s, t] = 0.5; // Initial Q value
        delta[s, t] = 0; // Initial prediction error
      } else if (choice[s, t] > -1) {
        // Calculate prediction error
        delta[s, t] = rew[s, t] - Q[s, t-1];
        // Update Q values based on alpha and prediction error
        Q[s, t] = Q[s, t-1] + alpha[s] * delta[s, t];
      } else {
        // Carry forward the Q values for missed trials
        Q[s, t] = Q[s, t-1];
        delta[s, t] = 0;
      }
    }
  }
}

model {
  // Priors
  a1 ~ cauchy(0, 5);
  a2 ~ cauchy(0, 5);
  beta ~ normal(0, 1);

  // Likelihood
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] > -1) { // Ensure the trial was not missed
        int choice_prob = choice[s, t];
        choice_prob ~ bernoulli_logit(beta * Q[s, t]);
      }
    }
  }
}
