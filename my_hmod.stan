data {
  int<lower=1> NS; // Number of subjects
  int<lower=1> total_observations; // Total number of trial observations across all subjects
  array[NS] int end_index; // End index of trials for each subject
  array[total_observations] int<lower=-1, upper=1> rew; // Rewards: 1 = positive, -1 = negative, 0 = missed trial
  array[total_observations] int<lower=0, upper=1> choice_cat; // Choices between categories: 1 for A, 0 for B, -1 for missed
}

parameters {
  real<lower=0> a1; // Shape parameter 1 for beta distribution of alpha
  real<lower=0> a2; // Shape parameter 2 for beta distribution of alpha
  vector[2] b_mean; // Mean of beta coefficients across subjects
  vector<lower=0>[2] b_sd; // Standard deviation of beta coefficients across subjects
  array[NS] real<lower=0, upper=1> alpha; // Individual learning rates
  array[NS] vector[2] beta; // Individual decision parameters for each subject
}

transformed parameters {
  array[total_observations, 2] real Q; // Q values for categories A and B
  array[total_observations] real delta; // Prediction errors

  // Initialize Q values and prediction errors
  for (int i = 1; i <= total_observations; i++) {
    Q[i, 1] = 0.5; // Initialize Q for category A
    Q[i, 2] = 0.5; // Initialize Q for category B
    delta[i] = 0;  // Initialize prediction error
  }

  // Update Q values based on trials and choices
  int obs_idx = 1;
  for (int s = 1; s <= NS; s++) {
    for (int idx = obs_idx; idx <= end_index[s]; idx++) {
      if (choice_cat[idx] != -1 && idx > obs_idx) {
        int choice_index = choice_cat[idx] + 1; // Convert choice to index
        delta[idx - 1] = rew[idx - 1] - Q[idx - 1, choice_index];
        Q[idx, choice_index] = Q[idx - 1, choice_index] + alpha[s] * delta[idx - 1];
        Q[idx, 3 - choice_index] = Q[idx - 1, 3 - choice_index]; // Unchosen remains the same
      }
    }
    obs_idx = end_index[s] + 1;
  }
}

model {
  a1 ~ cauchy(0, 5); // Prior for shape parameter of beta distribution
  a2 ~ cauchy(0, 5);
  b_mean ~ normal(0, 1); // Priors for group-level beta coefficients
  b_sd ~ cauchy(0, 1);

  alpha ~ beta(a1, a2); // Individual alphas follow a beta distribution
  for (int s = 1; s <= NS; s++) {
    beta[s] ~ normal(b_mean, b_sd); // Individual betas follow a normal distribution
  }

  // Likelihood of choices
  for (int i = 1; i <= total_observations; i++) {
    if (choice_cat[i] != -1) {
      choice_cat[i] ~ bernoulli_logit(beta[1] + beta[2] * (Q[i, 1] - Q[i, 2]));
    }
  }
}
