data {
  int<lower=1> NS; // Number of subjects
  int<lower=1> NT[NS]; // Number of trials per subject
  real<lower=-1, upper=1> rew[NS, max(NT)]; // Subject x trial reward, -1 for missed
  int<lower=-1, upper=2> choice[NS, max(NT)]; // Chosen category, 1 for A, 2 for B, -1 for missed
  int<lower=-1, upper=1> choice_cat[NS, max(NT)]; // 1=chose category A, 0=chose category B, -1 for missed
}

transformed data {
  int N; // Total number of non-missed trials
  N = 0;
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] > -1) {
        N = N + 1;
      }
    }
  }
}

parameters {
  real<lower=0, upper=1> alpha; // Learning rate
  vector[2] b_mean; // Coefficients for choice modeling
}

transformed parameters {
  real Q[NS, max(NT), 2]; // Q values for categories A and B
  real delta[NS, max(NT)]; // Prediction errors
  
  // Initialize Q values and prediction errors
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      for (c in 1:2) {
        Q[s, t, c] = (t == 1) ? 0.5 : Q[s, t-1, c]; // Set initial Q values and carry forward
      }
      delta[s, t] = 0; // Initialize prediction errors
    }
  }
  
  // Update Q values and calculate prediction errors
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s, t] > -1) { // Check for valid choice
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
      if (choice[s, t] > -1) {
        choice_cat[s, t] ~ bernoulli_logit(b_mean[1] + b_mean[2] * (Q[s, t, 1] - Q[s, t, 2]));
      }
    }
  }
}

// Include generated quantities for posterior checks or additional outputs as needed
