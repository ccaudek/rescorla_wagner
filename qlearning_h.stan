data {
  int<lower = 0> N_trials;
  int<lower = 0> N_subj;
  int<lower = 0> N_arms;
  matrix[N_trials, N_arms] R;
  array[N_trials, N_subj] int action;
}
transformed data{
  int N_adj = 3;
  array[N_trials, N_subj] int response;
    for(i in 1:N_subj)
      for(n in 1:N_trials)
        response[n,i] = action[n,i] - 1;
}
parameters {
  real<lower = 0, upper = 1> alpha;
  real beta_0;
  real beta_1;
  vector<lower = 0>[N_adj] tau_u;
  matrix[N_adj, N_subj] z_u;
  cholesky_factor_corr[N_adj] L_u;
}
transformed parameters{
  matrix[N_subj, N_adj] u = (diag_pre_multiply(tau_u, L_u) * z_u)';
}
model {
  matrix[N_arms, N_subj] Q = rep_matrix(.5,N_arms, N_subj);
  matrix[N_trials, N_subj] Q_diff;
  vector[N_subj] log_lik;
  for(i in 1:N_subj){
    real alpha_i = inv_logit(logit(alpha) + u[i, 1]);
    for(t in 1:N_trials){
      Q_diff[t, i] = Q[2, i] - Q[1, i];
      Q[action[t, i], i] += alpha_i * 
                        (R[t, action[t, i]] - Q[action[t, i], i]);
    }
    log_lik[i] = bernoulli_logit_lpmf(response[ ,i] | 
              beta_0 + u[i, 2] + (beta_1 + u[i, 3]) * Q_diff[,i]);
  }
  target += beta_lpdf(alpha | 1, 1);
  target += normal_lpdf(beta_0 | 2, 5);
  target += normal_lpdf(beta_1 | 0, 1);
  target += normal_lpdf(tau_u | .1, .5)
            - N_adj * normal_lccdf(0 | .1, .5);
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  target += std_normal_lpdf(to_vector(z_u));
  target += sum(log_lik);
}
