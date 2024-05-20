data {
  int<lower=1> NS; // number of subjects
  int<lower=1> MT; // maximum number of trials
  int<lower=1> NC; // number of choices (2)
  array[NS] int NT; // number of trials per subject
  array[NS, MT] real<lower=-1, upper=1> rew; // subject x trial reward
  array[NS, MT] int choice; // chosen option
  array[NS, MT] int choice_two; // 1 = chose left, 0 = chose right
}

parameters {
  real mu_alpha_logit;
  real<lower=0> sigma_alpha_logit;
  vector[NS] alpha_logit_raw;
  
  real mu_inv_temp_logit;
  real<lower=0> sigma_inv_temp_logit;
  vector[NS] inv_temp_logit_raw;
}

transformed parameters {
  vector<lower=0, upper=1>[NS] alpha;
  vector<lower=0, upper=5>[NS] inv_temp;

  alpha = inv_logit(mu_alpha_logit + sigma_alpha_logit * alpha_logit_raw);
  inv_temp = 5 * inv_logit(mu_inv_temp_logit + sigma_inv_temp_logit * inv_temp_logit_raw);
}

model {
  // Wider Hyperpriors
  mu_alpha_logit ~ normal(0, 2);
  sigma_alpha_logit ~ cauchy(0, 5);
  mu_inv_temp_logit ~ normal(0, 2);
  sigma_inv_temp_logit ~ cauchy(0, 5);

  // Priors for subject-level parameters
  alpha_logit_raw ~ normal(0, 1);
  inv_temp_logit_raw ~ normal(0, 1);

  // Likelihood
  for (s in 1:NS) {
    for (t in 1:NT[s]) {
      if (choice[s,t] > 0) {
        choice_two[s,t] ~ bernoulli_logit(inv_temp[s] * (alpha[s] * rew[s,t]));
      }
    }
  }
}

