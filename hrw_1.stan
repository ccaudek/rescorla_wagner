data {
  int<lower=1> NS; // Number of subjects
  array[NS] int NT; // Number of trials per subject
  array[NS] vector[max(NT)] rew; // Rewards: 1 = positive, -1 = negative, 0 = missed trial (placeholder for now)
  array[NS, max(NT)] int choice; // Choices between categories: 1 for A, 2 for B, -1 for missed
  array[NS, max(NT)] int choice_cat; // Binary choices for categories: 1 for A, 0 for B, -1 for missed
}

transformed data {
  int max_trials = max(NT); // Maximum number of trials across all subjects
}

parameters {
  real alpha_mu_logit; // Hyperparameter for mean learning rate in logit space
  real<lower=0> alpha_sigma; // Hyperparameter for std learning rate
  vector[NS] alpha_raw; // Non-centered individual learning rates in logit space

  real beta_mu; // Hyperparameter for mean temperature
  real<lower=0> beta_sigma; // Hyperparameter for std temperature
  vector[NS] beta_raw; // Non-centered individual temperature parameters

  vector[2] b_mu; // Hyperparameter for mean coefficients for choice modeling
  vector<lower=0>[2] b_sigma; // Hyperparameter for std coefficients
  matrix[NS, 2] b_raw; // Non-centered individual coefficients for choice modeling
}

transformed parameters {
  vector<lower=0, upper=1>[NS] alpha; // Individual learning rates
  vector[NS] beta; // Individual temperature parameters
  matrix[NS, 2] b; // Individual coefficients for choice modeling

  array[NS, max_trials, 2] real Q; // Q values for categories A and B
  array[NS, max_trials] real delta; // Prediction errors

  // Reparameterize the individual parameters
  alpha = inv_logit(alpha_mu_logit + alpha_sigma * alpha_raw);
  beta = beta_mu + beta_sigma * beta_raw;
  for (s in 1:NS) {
    b[s] = to_row_vector(b_mu + b_sigma .* to_vector(b_raw[s]));
  }

  // Initialize Q values and prediction errors
  for (s in 1:NS) {
    for (t in 1:max_trials) {
      for (c in 1:2) {
        Q[s, t, c] = (t == 1 || choice[s, t] == -1) ? 0.5 : Q[s, t-1, c]; // Set initial Q values and carry forward
      }
      delta[s, t] = 0; // Initialize prediction errors
    }
  }

  // Update Q values and calculate prediction errors
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] != -1) { // Check for valid choice
        int chosen_category = choice[s, t];
        delta[s, t] = rew[s, t] - Q[s, t, chosen_category];

        // Update Q values for next trial
        if (t < NT[s]) {
          Q[s, t+1, chosen_category] += alpha[s] * delta[s, t];
        }
      }
    }
  }
}

model {
  // Hyperpriors
  alpha_mu_logit ~ normal(0, 1); // Prior for group mean learning rate (logit scale)
  alpha_sigma ~ normal(0, 1); // Prior for group std dev of learning rate
  beta_mu ~ normal(0, 1); // Prior for group mean inverse temperature
  beta_sigma ~ normal(0, 1); // Slightly more diffuse prior for group std dev of inverse temperature
  b_mu ~ normal(0, 5); // Prior for group mean coefficients
  b_sigma ~ normal(0, 2); // Prior for group std dev of coefficients

  // Priors for raw parameters
  alpha_raw ~ normal(0, 1); // Raw learning rates
  beta_raw ~ normal(0, 1); // Raw inverse temperatures
  to_vector(b_raw) ~ normal(0, 1); // Raw coefficients

  // Likelihood of choices
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] != -1) {
        choice_cat[s, t] ~ bernoulli_logit(beta[s] * (b[s, 1] + b[s, 2] * (Q[s, t, 1] - Q[s, t, 2])));
      }
    }
  }
}

generated quantities {
  vector[NS] alpha_transformed;
  vector[NS] beta_transformed;

  for (s in 1:NS) {
    alpha_transformed[s] = inv_logit(alpha_mu_logit + alpha_sigma * alpha_raw[s]);
    beta_transformed[s] = beta_mu + beta_sigma * beta_raw[s];
  }
}
