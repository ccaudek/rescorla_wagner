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

d <- real_data %>%
  dplyr::filter((ema_number < 11))

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


# fit <- mod$sample(
#   data = stan_data,
#   parallel_chains = 4
# )
# 
# fit <- mod$sample(
#   data = stan_data,
#   seed = 123,
#   chains = 4,
#   parallel_chains = 4,
#   iter_warmup = 1000,
#   iter_sampling = 1000,
#   max_treedepth = 15,
#   adapt_delta = 0.99
# )

# Check sampling diagnostics
fit$diagnostic_summary()







# Import the data
real_data <- rio::import("groundhog_hddmrl_data.csv")

# Convert user_id to a numeric factor
real_data$id <- as.numeric(as.factor(as.character(real_data$user_id)))

# Filter data for specific conditions
d <- real_data %>%
  dplyr::filter((ema_number < 11) & (id < 50))

d <- real_data %>%
  dplyr::filter((ema_number < 11))

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
  NC = 2, 
  NT = NT, 
  rew = rew, 
  choice = choice, 
  choice_two = choice_two, 
  stickiness_indicator = stickiness
)


# go to m3_h_06

# ------------------------------------------------------------------------------
mod <- cmdstan_model("m3_h_04.stan") # NC parametrization

# Run the pathfinder variational method
fit_pf <- mod$pathfinder(
  data = stan_data, 
  seed = 123
  )

# fit_pf$summary()

# Assuming you have already fitted the model and have fit_pf
posterior_draws <- fit_pf$draws()

# Convert to a data frame using the posterior package
posterior_df <- as_draws_df(posterior_draws)

# Extract and summarize alpha parameters
alpha_samples <- posterior_df %>% dplyr::select(starts_with("alpha["))
summary_alpha <- alpha_samples %>% summarise_draws()
alpha_means <- summary_alpha %>%
  dplyr::filter(str_detect(variable, "alpha\\[.*\\]")) %>%
  dplyr::select(variable, mean)
hist(alpha_means$mean, 
     main = "Histogram of Alpha Means", xlab = "Mean Alpha", 
     col = "lightcoral")

# Extract and summarize inverse temperature parameters
inv_temp_samples <- posterior_df %>% dplyr::select(starts_with("inv_temp["))
summary_inv_temp <- inv_temp_samples %>% summarise_draws()
inv_temp_means <- summary_inv_temp %>%
  dplyr::filter(str_detect(variable, "inv_temp\\[.*\\]")) %>%
  dplyr::select(variable, mean)
print(inv_temp_means)
hist(inv_temp_means$mean, 
     main = "Histogram of Inverse Temperature Means", 
     xlab = "Mean Inverse Temperature", col = "lightblue")

# Extract and summarize stickiness parameters
stickiness_samples <- posterior_df %>% dplyr::select(starts_with("stickiness_param["))
summary_stickiness <- stickiness_samples %>% summarise_draws()
stickiness_means <- summary_stickiness %>%
  dplyr::filter(str_detect(variable, "stickiness_param\\[.*\\]")) %>%
  dplyr::select(variable, mean)
print(stickiness_means)
hist(stickiness_means$mean, main = "Histogram of Stickiness Means", xlab = "Mean Stickiness", col = "lightgreen")

# Hyper-parameters -----

posterior_draws <- fit_pf$draws()

# Convert to a data frame using the posterior package
posterior_df <- as_draws_df(posterior_draws)

# Display column names to verify the presence of the required columns
colnames(posterior_df)

# Extract hyperparameters for alpha positive, alpha negative, inverse temperature, and stickiness
hyperparameters <- posterior_df %>%
  dplyr::select(matches("mu_alpha_logit\\[1\\]|mu_alpha_logit\\[2\\]|sigma_alpha_logit\\[1\\]|sigma_alpha_logit\\[2\\]|mu_inv_temp|sigma_inv_temp|mu_stickiness|sigma_stickiness"))

# Rename columns for easier plotting
colnames(hyperparameters) <- c("mu_alpha_pos_logit", "mu_alpha_neg_logit", "sigma_alpha_pos_logit", "sigma_alpha_neg_logit", 
                               "mu_inv_temp", "sigma_inv_temp", "mu_stickiness", "sigma_stickiness")

# Apply logistic transformation to the alpha hyperparameters on the logit scale
hyperparameters <- hyperparameters %>%
  mutate(
    mu_alpha_pos = 1 / (1 + exp(-mu_alpha_pos_logit)),
    mu_alpha_neg = 1 / (1 + exp(-mu_alpha_neg_logit)),
    sigma_alpha_pos = 1 / (1 + exp(-sigma_alpha_pos_logit)),
    sigma_alpha_neg = 1 / (1 + exp(-sigma_alpha_neg_logit))
  )

# Select the transformed hyperparameters for plotting
hyperparameters_transformed <- hyperparameters %>%
  dplyr::select(mu_alpha_pos, mu_alpha_neg, sigma_alpha_pos, sigma_alpha_neg, 
         mu_inv_temp, sigma_inv_temp, mu_stickiness, sigma_stickiness)

# Melt the data frame for ggplot2
hyperparameters_long <- hyperparameters_transformed %>%
  pivot_longer(cols = everything(), names_to = "parameter", values_to = "value")

# Plot density plots for each hyperparameter
ggplot(hyperparameters_long, aes(x = value, fill = parameter)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ parameter, scales = "free", ncol = 2) +
  theme_minimal() +
  labs(title = "Posterior Distributions of Transformed Hyperparameters", x = "Value", y = "Density") +
  scale_x_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 5)) +
  scale_x_continuous(limits = c(0, max(hyperparameters_long$value)), breaks = scales::pretty_breaks(n = 5))


## No stickiness parameter -----------------------------------------------------

# It seems to be the best one.

mod <- cmdstan_model("m3_h_05.stan") 

# Run the pathfinder variational method
fit_pf <- mod$pathfinder(
  data = stan_data, 
  seed = 123
)


# Assuming you have already fitted the model and have fit_pf
posterior_draws <- fit_pf$draws()

# Convert to a data frame using the posterior package
posterior_df <- as_draws_df(posterior_draws)

# Extract and summarize alpha parameters
alpha_samples <- posterior_df %>% dplyr::select(starts_with("alpha["))
summary_alpha <- alpha_samples %>% summarise_draws()

# Filter and separate alpha positive and alpha negative parameters
alpha_means <- summary_alpha %>%
  dplyr::filter(str_detect(variable, "alpha\\[.*\\]")) %>%
  dplyr::select(variable, mean)

# Separate alpha positive and alpha negative
alpha_pos_means <- alpha_means %>%
  dplyr::filter(str_detect(variable, "alpha\\[.*,1\\]"))

alpha_neg_means <- alpha_means %>%
  dplyr::filter(str_detect(variable, "alpha\\[.*,2\\]"))

# Plot histograms for alpha positive
hist(alpha_pos_means$mean, 
     main = "Histogram of Alpha Positive Means", 
     xlab = "Mean Alpha Positive", 
     col = "lightcoral",
     xlim = c(0, 1))

# Plot histograms for alpha negative
hist(alpha_neg_means$mean, 
     main = "Histogram of Alpha Negative Means", 
     xlab = "Mean Alpha Negative", 
     col = "lightblue",
     xlim = c(0, 1))



# Extract and summarize inverse temperature parameters
inv_temp_samples <- posterior_df %>% dplyr::select(starts_with("inv_temp["))
summary_inv_temp <- inv_temp_samples %>% summarise_draws()
inv_temp_means <- summary_inv_temp %>%
  dplyr::filter(str_detect(variable, "inv_temp\\[.*\\]")) %>%
  dplyr::select(variable, mean)
print(inv_temp_means)
hist(inv_temp_means$mean, main = "Histogram of Inverse Temperature Means", xlab = "Mean Inverse Temperature", col = "lightblue")


# Hyper-parameters -----

posterior_draws <- fit_pf$draws()

# Convert to a data frame using the posterior package
posterior_df <- as_draws_df(posterior_draws)

# Display column names to verify the presence of the required columns
colnames(posterior_df)

# Extract hyperparameters for alpha positive, alpha negative, inverse temperature, and stickiness
hyperparameters <- posterior_df %>%
  dplyr::select(matches("mu_alpha_logit\\[1\\]|mu_alpha_logit\\[2\\]|sigma_alpha_logit\\[1\\]|sigma_alpha_logit\\[2\\]|mu_inv_temp|sigma_inv_temp"))

# Rename columns for easier plotting
colnames(hyperparameters) <- c("mu_alpha_pos_logit", "mu_alpha_neg_logit", "sigma_alpha_pos_logit", "sigma_alpha_neg_logit", 
                               "mu_inv_temp", "sigma_inv_temp")

# Apply logistic transformation to the alpha hyperparameters on the logit scale
hyperparameters <- hyperparameters %>%
  mutate(
    mu_alpha_pos = 1 / (1 + exp(-mu_alpha_pos_logit)),
    mu_alpha_neg = 1 / (1 + exp(-mu_alpha_neg_logit)),
    sigma_alpha_pos = 1 / (1 + exp(-sigma_alpha_pos_logit)),
    sigma_alpha_neg = 1 / (1 + exp(-sigma_alpha_neg_logit))
  )

# Select the transformed hyperparameters for plotting
hyperparameters_transformed <- hyperparameters %>%
  dplyr::select(mu_alpha_pos, mu_alpha_neg, sigma_alpha_pos, sigma_alpha_neg, 
                mu_inv_temp, sigma_inv_temp)

# Melt the data frame for ggplot2
hyperparameters_long <- hyperparameters_transformed %>%
  pivot_longer(cols = everything(), names_to = "parameter", values_to = "value")

# Plot density plots for each hyperparameter
ggplot(hyperparameters_long, aes(x = value, fill = parameter)) +
  geom_density(alpha = 0.6) +
  facet_wrap(~ parameter, scales = "free", ncol = 2) +
  theme_minimal() +
  labs(title = "Posterior Distributions of Transformed Hyperparameters", x = "Value", y = "Density") +
  scale_x_continuous(limits = c(0, 1), breaks = scales::pretty_breaks(n = 5)) +
  scale_x_continuous(limits = c(0, max(hyperparameters_long$value)), breaks = scales::pretty_breaks(n = 5))


# KEEP THIS ONE ! ! ! !






# eof --------------------------------------------------------------------------



# Single learning rate parameter model -----------------------------------------

mod <- cmdstan_model("m3_h_07.stan") 

# Run the pathfinder variational method
fit_pf <- mod$pathfinder(
  data = stan_data, 
  seed = 123
)


# Assuming you have already fitted the model and have fit_pf
posterior_draws <- fit_pf$draws()

# Convert to a data frame using the posterior package
posterior_df <- as_draws_df(posterior_draws)

# Extract hyperparameters for alpha and inverse temperature
hyperparameters <- posterior_df %>%
  dplyr::select(matches("mu_alpha_logit|sigma_alpha_logit|mu_inv_temp_log|sigma_inv_temp_log"))

# Rename columns for easier plotting
colnames(hyperparameters) <- c("mu_alpha_logit", "sigma_alpha_logit", 
                               "mu_inv_temp_log", "sigma_inv_temp_log")

# Apply logistic transformation to the alpha hyperparameters on the logit scale
hyperparameters <- hyperparameters %>%
  mutate(
    mu_alpha = 1 / (1 + exp(-mu_alpha_logit)),
    sigma_alpha = 1 / (1 + exp(-sigma_alpha_logit)),
    mu_inv_temp = exp(mu_inv_temp_log),
    sigma_inv_temp = exp(sigma_inv_temp_log)
  )

# Select the transformed hyperparameters for plotting
hyperparameters_transformed <- hyperparameters %>%
  dplyr::select(mu_alpha, sigma_alpha, mu_inv_temp, sigma_inv_temp)

# Plot posterior distributions for each hyperparameter
plot_list <- list()

plot_list[[1]] <- ggplot(hyperparameters_transformed, aes(x = mu_alpha)) +
  geom_histogram(binwidth = 0.01, fill = "lightcoral", color = "black") +
  labs(title = "Posterior Distribution of mu_alpha", x = "mu_alpha", y = "Frequency") +
  xlim(0, 1)

plot_list[[2]] <- ggplot(hyperparameters_transformed, aes(x = sigma_alpha)) +
  geom_histogram(binwidth = 0.01, fill = "lightblue", color = "black") +
  labs(title = "Posterior Distribution of sigma_alpha", x = "sigma_alpha", y = "Frequency") +
  xlim(0, 1)

plot_list[[3]] <- ggplot(hyperparameters_transformed, aes(x = mu_inv_temp)) +
  geom_histogram(binwidth = 0.1, fill = "lightgreen", color = "black") +
  labs(title = "Posterior Distribution of mu_inv_temp", x = "mu_inv_temp", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$mu_inv_temp))

plot_list[[4]] <- ggplot(hyperparameters_transformed, aes(x = sigma_inv_temp)) +
  geom_histogram(binwidth = 0.1, fill = "lightpink", color = "black") +
  labs(title = "Posterior Distribution of sigma_inv_temp", x = "sigma_inv_temp", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$sigma_inv_temp))

# Print plots
for (plot in plot_list) {
  print(plot)
}


alpha_samples <- posterior_df %>% dplyr::select(starts_with("alpha["))
inv_temp_samples <- posterior_df %>% dplyr::select(starts_with("inv_temp["))

summary_alpha <- alpha_samples %>% summarise_draws()
summary_inv_temp <- inv_temp_samples %>% summarise_draws()

alpha_means <- summary_alpha %>%
  dplyr::filter(str_detect(variable, "alpha\\[.*\\]")) %>%
  dplyr::select(variable, mean)

inv_temp_means <- summary_inv_temp %>%
  dplyr::filter(str_detect(variable, "inv_temp\\[.*\\]")) %>%
  dplyr::select(variable, mean)



# Plot for Alpha Parameters
ggplot(alpha_means, aes(x = mean)) +
  geom_histogram(binwidth = 0.02, fill = "lightcoral", color = "black", alpha = 0.7) +
  labs(title = "Posterior Distribution of Alpha Parameters", x = "Alpha", y = "Frequency") +
  xlim(0, 1)

# Plot for Inverse Temperature Parameters
ggplot(inv_temp_means, aes(x = mean)) +
  geom_histogram(binwidth = 0.1, fill = "lightblue", color = "black", alpha = 0.7) +
  labs(title = "Posterior Distribution of Inverse Temperature Parameters", x = "Inverse Temperature", y = "Frequency") +
  xlim(0, 5)










# Hyper-parameters -------------------------------------------------------------


# Extract hyperparameters for alpha, inverse temperature, and stickiness
hyperparameters <- posterior_df %>%
  dplyr::select(matches("mu_alpha_logit|sigma_alpha_logit|mu_inv_temp|sigma_inv_temp|mu_stickiness|sigma_stickiness"))

# Rename columns for easier plotting
colnames(hyperparameters) <- c("mu_alpha_logit", "sigma_alpha_logit", 
                               "mu_inv_temp", "sigma_inv_temp", 
                               "mu_stickiness", "sigma_stickiness")

# Apply logistic transformation to the alpha hyperparameters on the logit scale
hyperparameters <- hyperparameters %>%
  mutate(
    mu_alpha = 1 / (1 + exp(-mu_alpha_logit)),
    sigma_alpha = 1 / (1 + exp(-sigma_alpha_logit))
  )

# Select the transformed hyperparameters for plotting
hyperparameters_transformed <- hyperparameters %>%
  dplyr::select(mu_alpha, sigma_alpha, mu_inv_temp, sigma_inv_temp, mu_stickiness, sigma_stickiness)

# Plot posterior distributions for each hyperparameter
plot_list <- list()

plot_list[[1]] <- ggplot(hyperparameters_transformed, aes(x = mu_alpha)) +
  geom_histogram(binwidth = 0.01, fill = "lightcoral", color = "black") +
  labs(title = "Posterior Distribution of mu_alpha", x = "mu_alpha", y = "Frequency") +
  xlim(0, 1)

plot_list[[2]] <- ggplot(hyperparameters_transformed, aes(x = sigma_alpha)) +
  geom_histogram(binwidth = 0.01, fill = "lightblue", color = "black") +
  labs(title = "Posterior Distribution of sigma_alpha", x = "sigma_alpha", y = "Frequency") +
  xlim(0, 1)

plot_list[[3]] <- ggplot(hyperparameters_transformed, aes(x = mu_inv_temp)) +
  geom_histogram(binwidth = 0.1, fill = "lightgreen", color = "black") +
  labs(title = "Posterior Distribution of mu_inv_temp", x = "mu_inv_temp", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$mu_inv_temp))

plot_list[[4]] <- ggplot(hyperparameters_transformed, aes(x = sigma_inv_temp)) +
  geom_histogram(binwidth = 0.1, fill = "lightpink", color = "black") +
  labs(title = "Posterior Distribution of sigma_inv_temp", x = "sigma_inv_temp", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$sigma_inv_temp))

plot_list[[5]] <- ggplot(hyperparameters_transformed, aes(x = mu_stickiness)) +
  geom_histogram(binwidth = 0.1, fill = "lightyellow", color = "black") +
  labs(title = "Posterior Distribution of mu_stickiness", x = "mu_stickiness", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$mu_stickiness))

plot_list[[6]] <- ggplot(hyperparameters_transformed, aes(x = sigma_stickiness)) +
  geom_histogram(binwidth = 0.1, fill = "lightpurple", color = "black") +
  labs(title = "Posterior Distribution of sigma_stickiness", x = "sigma_stickiness", y = "Frequency") +
  xlim(0, max(hyperparameters_transformed$sigma_stickiness))

# Print plots
for (plot in plot_list) {
  print(plot)
}


