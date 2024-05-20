data {
  int<lower=0> N_trials;
  array[N_trials] int action;
  int N_arms;
  matrix[N_trials,N_arms] R;
}
transformed data {
  array[N_trials] int response;
  for(n in 1:N_trials)
    response[n] = action[n] - 1;
}
parameters {
  real<lower = 0, upper = 1> alpha;
  real beta_0;
  real beta_1;
}
model {
  vector[N_arms] Q = [.5, .5]';
  vector[N_trials] Q_diff;
  for(t in 1:N_trials){
    Q_diff[t] = Q[2] - Q[1];
    Q[action[t]] += alpha * (R[t, action[t]] - Q[action[t]]);
  }
  target += beta_lpdf(alpha | 1, 1);
  target += normal_lpdf(beta_0 | 2, 5);
  target += normal_lpdf(beta_1 | 0, 1);
  target += bernoulli_logit_lpmf(response | beta_0 + beta_1 * Q_diff);
}
