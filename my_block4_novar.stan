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
  real<lower=0, upper=1> alpha; // Learning rate
  vector[2] b_mean; // Coefficients for choice modeling
}

transformed parameters {
  array[NS, max_trials, 2] real Q; // Q values for categories A and B
  array[NS, max_trials] real delta; // Prediction errors
  
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
          Q[s, t+1, chosen_category] += alpha * delta[s, t];
        }
      }
    }
  }
}

model {
  alpha ~ beta(2, 2); // Prior on learning rate
  b_mean ~ normal(0, 5); // Priors on coefficients
  
  // Likelihood of choices
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] != -1) {
        choice_cat[s, t] ~ bernoulli_logit(b_mean[1] + b_mean[2] * (Q[s, t, 1] - Q[s, t, 2]));
      }
    }
  }
}

// Include generated quantities for posterior checks or additional outputs as needed
