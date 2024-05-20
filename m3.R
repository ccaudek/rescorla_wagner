library(tidyr) # To pivot the data
library(tidyverse)
library(purrr) # List manipulation
library(ggplot2) # Nice plots
library(extraDistr) # More distributions
library(MASS) # Tu use the multinormal distribution
library(cmdstanr) # Lightweight Stan interface
library(bayesplot) # Nice Bayesian plots
set.seed(123)



# Import the data
real_data <- rio::import("groundhog_hddmrl_data.csv")

# Convert user_id to a numeric factor
real_data$id <- as.numeric(as.factor(as.character(real_data$user_id)))

# Filter data for specific conditions
d <- real_data %>%
  dplyr::filter((ema_number < 11) & (id > 21 & id < 25))

# Add block variable
d <- d %>%
  mutate(block = ceiling(trial / 30))

# Extract unique subject IDs and calculate the number of subjects
unique_subjects <- unique(d$user_id)
NS <- length(unique_subjects)

# Get the maximum number of trials for any subject
max_trials_per_subject <- d %>%
  group_by(user_id) %>%
  summarise(trials = n()) %>%
  ungroup()

MT <- max(max_trials_per_subject$trials)

# Create a vector to hold the number of trials per subject
NT <- max_trials_per_subject$trials

# Initialize the matrices for rew, response, choice_two, stickiness, and block
rew <- matrix(0, nrow = NS, ncol = MT)
response <- matrix(0, nrow = NS, ncol = MT)
choice_two <- matrix(0, nrow = NS, ncol = MT)
stickiness <- matrix(0, nrow = NS, ncol = MT)
block <- matrix(0, nrow = NS, ncol = MT)

# Fill the matrices
for (i in 1:NS) {
  subj_data <- d %>%
    dplyr::filter(user_id == unique_subjects[i])
  n_trials <- nrow(subj_data)
  
  if (n_trials > 0) {
    # Ensure response values are valid (1 or 0)
    subj_data$response <- ifelse(!subj_data$response %in% c(1, 0), 1, subj_data$response)
    
    # Assign values to matrices, ensuring to not exceed the bounds
    rew[i, 1:n_trials] <- subj_data$feedback  # Assuming feedback is the reward
    response[i, 1:n_trials] <- subj_data$response
    choice_two[i, 1:n_trials] <- ifelse(subj_data$response == 1, 1, 0)
    
    if (n_trials > 1) {
      stickiness[i, 2:n_trials] <- ifelse(subj_data$response[1:(n_trials - 1)] == 1, 1, -1)
      stickiness[i, 1] <- 0  # No previous response for the first trial
    } else {
      stickiness[i, 1] <- 0
    }
    
    # Create the choice matrix based on response
    choice <- choice_two  # If you need the choice matrix to reflect a specific format, adjust accordingly
  }
}

# Prepare the list for Stan model
stan_data <- list(
  NS = NS,
  MT = MT,
  NC = 2,  # Number of choices
  NT = as.array(NT),
  rew = rew,
  choice = choice,
  choice_two = choice_two,
  stickiness = stickiness
)


mod <- cmdstan_model("m3_inf_var_03.stan")

fit <- mod$sample(
  data = stan_data,
  parallel_chains = 4
)


#' alpha[1,1] to alpha[5,1]: 
#' These represent the learning rates for positive prediction errors for 
#' subjects 1 to 5, respectively. In the context of the Rescorla-Wagner 
#' model with dual learning rates, these parameters determine how much the 
#' Q-value is updated when the observed reward is greater than the expected 
#' reward (i.e., when the prediction error is positive).
#' 
#' alpha[1,2] to alpha[5,2]: 
#' These represent the learning rates for negative prediction errors for 
#' subjects 1 to 5, respectively. These parameters determine how much the 
#' Q-value is updated when the observed reward is less than the expected 
#' reward (i.e., when the prediction error is negative).
#' 
#' beta[1,1] to beta[5,1]: 
#' These parameters represent the baseline preference for choosing the 
#' left option (or choice 1) over the right option (or choice 2) for 
#' subjects 1 to 5, respectively. This can be seen as a bias term in the 
#' logistic regression model.
#' 
#' beta[1,2] to beta[5,2]: 
#' These parameters represent the weight given to the difference in 
#' Q-values between the two choices for subjects 1 to 5, respectively. 
#' This term indicates how much the expected value difference between 
#' choices influences the decision-making process.
#' 
#' beta[1,3] to beta[5,3]: 
#' These parameters represent the weight given to the stickiness effect 
#' (or persistence) in the model for subjects 1 to 5, respectively. The 
#' stickiness parameter accounts for the tendency of subjects to repeat 
#' their previous choices.
#' 
#' alpha[s,1]: Learning rate for positive prediction errors for subject s.
#' alpha[s,2]: Learning rate for negative prediction errors for subject s.
#' beta[s,1]: Baseline preference for choosing left (or choice 1) for subject s.
#' beta[s,2]: Influence of Q-value difference on choice for subject s.
#' beta[s,3]: Stickiness parameter indicating persistence in choices for subject s.
#' 
#' inv_temp[s]: This is the inverse temperature parameter for subject s. 
#' It controls how deterministic the subject's choices are based on the 
#' Q-value differences. A higher value of inv_temp[s] implies more 
#' deterministic choices, where the subject strongly prefers the option with 
#' a higher Q-value. A lower value of inv_temp[s] implies more random choices, 
#' where the Q-value differences have less influence on the subject's decisions.


out <- fit$summary()
out$variable[1:31]


draws <- fit$draws(format = "df")
mcmc_hist(draws, pars='alpha[8,1]') # alpha[s, 1] positive alpha
mcmc_hist(draws, pars='alpha[4,2]') # alpha[s, 2] negative alpha
mcmc_hist(draws, pars='beta[5,3]') # stickiness
mcmc_hist(draws, pars='inv_temp[4]') # beta

# This converges. 
# But the parameters are all equal and are determined by the prior.

# Hierarchical model -----------------------------------------------------------

mod <- cmdstan_model("m3_h_02.stan") # NC parametrization

fit <- mod$sample(
  data = stan_data,
  parallel_chains = 4
)

fit <- mod$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  max_treedepth = 15,
  adapt_delta = 0.99
)

# Check sampling diagnostics
fit$diagnostic_summary()



# Run the variational method
fit_pf <- mod$variational(
  data = stan_data, 
  seed = 123
  )



# ------------------------------------------------------------------------------
# Stickiness to the prevailing choice to the first 15 trials.
# This does not work!

# Add block variable in the real_data
real_data <- real_data %>%
  mutate(block = ceiling(trial / 30))

# Extract unique subject IDs
unique_subjects <- unique(real_data$user_id)
NS <- length(unique_subjects)
MT <- max(real_data %>% group_by(user_id) %>% summarise(trials = n()) %>% pull(trials))

# Initialize arrays
rew <- matrix(0, nrow = NS, ncol = MT)
choice <- matrix(0, nrow = NS, ncol = MT)
choice_two <- matrix(0, nrow = NS, ncol = MT)
stickiness <- matrix(0, nrow = NS, ncol = MT)
block <- matrix(0, nrow = NS, ncol = MT)
NT <- integer(NS)

# Fill arrays with data
for (i in 1:NS) {
  subj_data <- real_data %>% filter(user_id == unique_subjects[i])
  n_trials <- nrow(subj_data)
  
  rew[i, 1:n_trials] <- subj_data$reward
  choice[i, 1:n_trials] <- subj_data$choice
  choice_two[i, 1:n_trials] <- ifelse(subj_data$choice == 1, 1, 0)
  stickiness[i, 1:n_trials] <- ifelse(lag(subj_data$choice, default = subj_data$choice[1]) == 1, 1, -1)
  block[i, 1:n_trials] <- subj_data$block
  NT[i] <- n_trials
}

# Create the list for stan
stan_data <- list(
  NS = NS,
  MT = MT,
  NC = 2,
  NT = NT,
  rew = rew,
  choice = choice,
  choice_two = choice_two,
  stickiness = stickiness,
  block = block
)

mod <- cmdstan_model("m3_inf_var_04.stan")

fit <- mod$sample(
  data = stan_data,
  parallel_chains = 4
)

out <- fit$summary()
out$variable[1:31]


draws <- fit$draws(format = "df")
mcmc_hist(draws, pars='alpha[1,2]')
mcmc_hist(draws, pars='beta[5,3]')
mcmc_hist(draws, pars='inv_temp[1]')

