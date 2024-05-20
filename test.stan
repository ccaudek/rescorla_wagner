data {
  int<lower=1> NS; // Number of subjects
  int<lower=1> total_observations; // Total number of trial observations across all subjects
  array[NS] int end_index; // End index of trials for each subject
  array[total_observations] int<lower=-1, upper=1> rew; // Rewards: 1 = positive, -1 = negative
  array[total_observations] int<lower=0, upper=1> choice_cat; // Choices between categories: 1 for A, 0 for B
}

parameters {
  array[NS] real<lower=0, upper=1> alpha; // Learning rates for each subject
  array[NS] vector[2] beta; // Coefficients for logistic regression for each subject
}

model {
  // Declare and initialize local variables for Q-values and prediction errors
  array[total_observations, 2] real Q; // Q values for categories A and B
  array[total_observations] real delta; // Prediction errors

  // Initialize Q values and prediction errors
  for ( i in 1:total_observations) {
    Q[i, 1] = 0.5; // Initialize Q for category A
    Q[i, 2] = 0.5; // Initialize Q for category B
    delta[i] = 0;  // Initialize prediction error
  }

  // Priors for learning rates and coefficients
  for ( s in 1:NS) {
    alpha[s] ~ beta(1, 1); // Assuming a uniform prior over the interval [0,1]
    beta[s] ~ normal(0, 1); // Assuming coefficients are normally distributed
  }

  int obs_idx = 1;
  int choice_index;
  real eta;

  // Model
  for ( s in 1:NS) {
    for ( idx in obs_idx:end_index[s]) {
      if (choice_cat[idx] != -1) {
        choice_index = choice_cat[idx] + 1; // Convert choice to index
        eta = beta[s][1] + beta[s][2] * (Q[idx - 1, 1] - Q[idx - 1, 2]); // Calculate logit
        choice_cat[idx] ~ bernoulli_logit(eta);
        delta[idx - 1] = rew[idx - 1] - Q[idx - 1, choice_index];
        Q[idx, choice_index] = Q[idx - 1, choice_index] + alpha[s] * delta[idx - 1];
        Q[idx, 3 - choice_index] = Q[idx - 1, 3 - choice_index]; // Unchosen remains the same
      }
    }
    obs_idx = end_index[s] + 1;
  }
}
