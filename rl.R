#### Overview ####
# Associated project: Groundhog Day  
# Script purpose: To estimate the Rescorla-Wagner parameters from EMA data
# Created by: Corrado Caudek
# Version: 2024-05-19
# Date modified: Mon May 20 06:13:32 2024
# Notes: In progress


#### Workspace setup ####

library(tidyr) # To pivot the data
library(tidyverse)
library(purrr) # List manipulation
library(ggplot2) # Nice plots
library(extraDistr) # More distributions
library(MASS) # Tu use the multinormal distribution
library(cmdstanr) # Lightweight Stan interface
library(bayesplot) # Nice Bayesian plots
set.seed(123)


#### Verify if the model can accurately capture the simulated parameters ####

# (01) Set parameters

alpha <- 0.1  # Learning rate
beta <- 2  # Inverse temperature (controls the randomness in choice)
n_trials <- 200  # Number of trials

# (02) Initialize variables

V <- c(0.5, 0.5)  # Initial values for options 1 and 2
choices <- numeric(n_trials)  # Vector to store choices
rewards <- numeric(n_trials)  # Vector to store rewards

set.seed(1234)  # For reproducibility

# (03) Simulate the data

for (i in 1:n_trials) {
  # Calculate probabilities of choosing each option using a softmax function
  prob <- exp(beta * V) / sum(exp(beta * V))
  
  # Simulate choice based on the probabilities
  choices[i] <- ifelse(runif(1) < prob[1], 1, 2)
  
  # Define reward probabilities for each option (these could be modified as needed)
  reward_prob <- ifelse(choices[i] == 1, 0.8, 0.2)  # Option 1 has 80% chance of reward, option 2 has 20%
  
  # Simulate reward based on chosen option
  rewards[i] <- ifelse(runif(1) < reward_prob, 1, -1)
  
  # Update the value of the chosen option using the Rescorla-Wagner rule
  V[choices[i]] <- V[choices[i]] + alpha * (rewards[i] - V[choices[i]])
}

# (04) Create a data frame for analysis or plotting

data <- data.frame(Trial = 1:n_trials, Choice = choices, Reward = rewards)

# (05) Generate the input list for cmdstan 

stan_list <- list(
  nTrials = dim(data)[1],
  choice = data$Choice,
  reward = data$Reward
)

# (06) Compile model

m_ql <- cmdstan_model("my_1st_rw.stan")

fit_ql <- m_ql$sample(
  data = stan_list,
  parallel_chains = 4
)

fit_ql

mcmc_recover_hist(fit_ql$draws(
  c("alpha", "tau")
),
true = c(alpha, beta)
)


#### Data import ####

# (01) Read real data

real_data <- rio::import("groundhog_hddmrl_data.csv")

# (02) Data wrangling

real_data$choice <- ifelse(real_data$response == 1, 1, 2)
real_data$reward <- ifelse(real_data$response == 1, 1, -1)

unique(real_data$user_id)
# [1] 3200947930  3208693225  3246292649  3247822380  3248648540  3270624644 
# [7] 3270947539  3271273161  3272161067  3273716057  3274685227  3279312977 
# [13] 3280004109  3280632529  3282753348  3282923087  3285486203  3294886377 
# [19] 3295682008  330266513   3311032316  3311128391  3311506647  3311815277 
# [25] 3311919255  3312571418  3313224759  3313233935  3313450012  3314389060 
# [31] 3314455802  3314594006  3314994466  3316198240  3317545421  3317606566 
# [37] 3317719173  3318067046  3318207859  3331208683  3333087407  3334297641 
# [43] 3334364444  3334542104  3334971564  3335607912  3336428135  3336846971 
# [49] 3336861167  3336930797  3337222747  3338029881  3338263250  3338296770 
# [55] 3338836285  3339582765  3339796982  3339880271  3341420106  3342177251 
# [61] 3342184754  3342380709  3342410340  3342493833  3343693017  3346121179 
# [67] 3347117521  3347443867  3347892623  3348106800  3349010634  3349104292 
# [73] 3356649726  33783670400 3381060433  3382161024  3382353432  3383061603 
# [79] 3383928947  3384230549  3385930632  3386008212  3387752052  3388833022 
# [85] 3391467959  3391627654  3392659454  3392736399  3394125847  3394341908 
# [91] 3394466675  3396144039  3396926438  3397350077  3398621330  3398964382 
# [97] 3401031360  3401270582  3401961300  3402664341  3403943604  3407230127 
# [103] 3420200692  3420547059  3420798804  3421843835  3421895411  3423734900 
# [109] 3427264062  3427335142  3427812776  3428976733  3450382042  3451263546 
# [115] 3452198846  3453303700  3453934331  3456058891  3456115892  3458476286 
# [121] 3458886355  3458983574  3459300091  3459372766  3460591283  3463723465 
# [127] 3465959424  3467014707  3469464166  3469730627  3472369691  3473918493 
# [133] 3474380614  3475648095  3477079323  3478294460  3483195353  3483232737 
# [139] 3483400286  3485956383  3487119790  3490815665  3491070201  3492366988 
# [145] 3492432569  3493200977  3493727344  3493865607  3494499091  3496276977 
# [151] 3496897554  3497824506  3505065648  3661166007  3661375646  3662125608 
# [157] 3662258943  3662879175  3662895706  3663738920  3663995281  3664300311 
# [163] 3664548042  3664904129  3665244090  3665345709  3665355105  3667000898 
# [169] 3667272818  3668615349  3669301160  3703103013  3703107268  3703258744 
# [175] 3703293103  3703403724  3771015565  3774219750  3792840489  3802426721 
# [181] 3806823362  3806950650  3807607123  3808941373  3809075395  3880596637 
# [187] 3881466599  3883816051  3888185906  3888281971  3890065938  3891205133 
# [193] 3891217212  3892305104  3895555996  3895730674  3896211241  3911072456 
# [199] 3913468055  3920062595  3920224683  3920921922  3921884140  3925461159 
# [205] 3927302099  3935500366  4389943231 

real_data$id = as.numeric(as.factor(as.character(real_data$user_id)))

# (03) Select one subject only

one_subject <- real_data |> 
  dplyr::filter(id == 7)

# TODO Add missing information!
# Rewards: Defined as `1` for positive, `-1` for negative, 
# and `0` for missed trials. This directly influences the calculation 
# of prediction errors (`delta`) and subsequent updates to the Q-values.

# Handling of Missed Trials: By checking if `choice[s, t] != -1`, 
# the script ensures that missed trials do not influence the calculations 
# or model likelihood.

stan_list <- list(
  nTrials = dim(one_subject)[1],
  choice = one_subject$choice,
  reward = one_subject$reward
)

# Use `alpha` as the learning rate and `b_mean` for the logistic regression 
# coefficients. The likelihood function uses the logistic regression model to 
# predict binary choices based on the computed Q-value differences between the 
# two categories.


fit_ql <- m_ql$sample(
  data = stan_list,
  parallel_chains = 4
)

fit_ql$summary()

draws <- fit_ql$draws(format = "df")
mcmc_hist(draws, pars='alpha') + xlim(c(0,1))
mcmc_hist(draws, pars='tau') + xlim(c(0,1))



m1_ql <- cmdstan_model("my_block4_novar.stan")



# subset_id <- c(3401961300) # runs only with a single subject
subset_id <- c(3401961300, 3270947539,  3271273161,  3272161067)

# Filter and prepare the dataset
small_subset <- real_data %>% 
  filter(id < 6) %>%
  arrange(user_id, trial) %>%
  mutate(
    choice_cat = case_when(
      choice == 1 ~ 1,
      choice == 2 ~ 0,
      TRUE ~ -1
    ),
    trial_index = row_number()
  )

# Calculate number of trials per subject
trial_counts <- small_subset %>%
  group_by(user_id) %>%
  summarise(
    n_trials = n(),
    .groups = 'drop'
  )

# Prepare the data matrices
NT <- trial_counts$n_trials
max_trials <- max(NT)

# Function to pad the vector for matrix preparation
pad_vector <- function(x, max_length, pad_value = NA) {
  length(x) <- max_length
  x[is.na(x)] <- pad_value
  x
}

# Create matrices for rewards, choices, and choice categories
create_matrix <- function(data, variable, pad_value) {
  data %>%
    group_by(user_id) %>%
    summarise(
      data_list = list(pad_vector(get(variable), max_trials, pad_value)),
      .groups = 'drop'
    ) %>%
    pull(data_list) %>%
    simplify2array() %>%
    aperm(c(2, 1))  # Transpose to match Stan's expectations
}

rew_matrix <- create_matrix(small_subset, "reward", 0)
choice_matrix <- create_matrix(small_subset, "choice", -1)
choice_cat_matrix <- create_matrix(small_subset, "choice_cat", -1)

# Prepare the list for Stan
stan_list <- list(
  NS = length(NT),
  NT = NT,
  rew = rew_matrix,
  choice = choice_matrix,
  choice_cat = choice_cat_matrix
)

mod <- cmdstan_model("hrw_1.stan")

fit <- mod$sample(
  data = stan_list,
  parallel_chains = 4
)

# fit_ql$summary()

draws <- fit$draws(format = "df")
mcmc_hist(draws, pars='alpha_transformed[5]')
mcmc_hist(draws, pars='beta_transformed[3]')






temps <- fit_ql$draws(format = "df") |>
  as_tibble() |>
  dplyr::select(starts_with("b_mean["))

mcmc_areas(temps) + xlab('Temperature')


# Hierarchical model -----------------------------------------------------------


# First, ensure the data is sorted and grouped as needed
small_subset <- small_subset %>%
  arrange(user_id, trial)

# Calculate the maximum number of trials for each subject
max_trials_per_user <- small_subset %>%
  group_by(user_id) %>%
  summarize(max_trial = max(trial, na.rm = TRUE)) %>%
  ungroup()

# Ensure each user has at least 120 trials, or their maximum if greater
max_trials_per_user$max_trial <- pmax(120, max_trials_per_user$max_trial)

# Join back to the original dataset to get a filter mask
small_subset <- small_subset %>%
  left_join(max_trials_per_user, by = "user_id") %>%
  group_by(user_id) %>%
  filter(row_number() <= max_trial) %>%
  ungroup() %>%
  select(-max_trial)  # Remove the extra column after filtering


# Count the number of trials for each user
trial_counts <- small_subset %>%
  group_by(user_id) %>%
  summarize(n_trials = n()) %>%
  ungroup()

# Ensure each subject has an entry for each trial up to their maximum
# and create the input for the Stan model
stan_data <- small_subset %>%
  group_by(user_id) %>%
  mutate(
    trial_id = row_number(),  # Ensures trials are sequentially numbered per user
    rew_adjusted = if_else(reward == -1, 0, 1),  # Convert rewards to 0/1 for missed/awarded
    choice_adjusted = if_else(choice_cat == 1, 1, 0)  # Convert choices to 1 for A, 0 for B
  ) %>%
  ungroup() %>%
  arrange(user_id, trial_id)

# Creating arrays for Stan
ns <- n_distinct(stan_data$user_id)  # number of subjects
nt <- max(trial_counts$n_trials)  # maximum number of trials

# Prepare the list for Stan
stan_list <- list(
  NS = ns,
  NT = rep(nt, ns),  # Assuming all subjects are filled to max trials
  MT = 300,
  rew = matrix(
    stan_data$rew_adjusted,
    nrow = ns,
    ncol = nt,
    byrow = TRUE
  ),
  choice = matrix(
    stan_data$choice_adjusted,
    nrow = ns,
    ncol = nt,
    byrow = TRUE
  )
)

# Check the structure
str(stan_list)

m1h_ql <- cmdstan_model("hierarchical1.stan")

fit_hql <- m1h_ql$sample(
  data = stan_list,
  parallel_chains = 4
)

fit_hql

# Compilo il modello gerarchico nella versione degli autori, 
# con la sintassi corretta.
m1h <- cmdstan_model("new_syntax_Block4.stan")

# ------------------------------------------------------------------------------
#' Compile the simplified Stan code, created by chatGPT, that is 
#' appropriate for my experimental design. In my case, there is only 
#' one condition (no multiple stimuli), and I have blocks of 30 trials 
#' with a minimum of 4 blocks and a maximum of 10 blocks per participant.
m2h <- cmdstan_model("my_hierarchical_1.stan")



subset_id <- c(3465959424,  3467014707,  3469464166,  3469730627,  
               3472369691,  3473918493)
subset_id <- sort(subset_id)
print(subset_id)


# Filter and prepare the dataset
small_subset <- real_data %>%
  filter(user_id %in% subset_id) %>%
  arrange(user_id, trial) %>%
  mutate(
    choice_cat = case_when(
      choice == 1 ~ 1,
      choice == 2 ~ 0,
      TRUE ~ -1
    ),
    trial_index = row_number()
  )

# Convert choice values for Stan compatibility
small_subset <- small_subset %>%
  mutate(
    choice = if_else(choice == 2, 0, choice)  # Convert 2 to 0 for right choice
  )

# Count the number of trials for each user
trial_counts <- small_subset %>%
  group_by(subj_idx) %>%
  summarize(n_trials = n()) %>%
  ungroup()

# Define constants
NC <- 2
K <- 2  # Assuming 2 coefficients for simplicity, adjust based on your actual model

# Prepare the data list for Stan
NS <- length(unique(small_subset$subj_idx))
MT <- max(trial_counts$n_trials)
reward <- matrix(-1, nrow = NS, ncol = MT)    # Default placeholder value -1
choice <- matrix(-1, nrow = NS, ncol = MT)    # Default placeholder value -1
choice_two <- matrix(-1, nrow = NS, ncol = MT) # Default placeholder value -1

# Actual subject indices
subj_indices <- unique(small_subset$subj_idx)

for (i in seq_along(subj_indices)) {
  subj_id <- subj_indices[i]
  subj_data <- small_subset %>% filter(subj_idx == subj_id)
  n_trials <- nrow(subj_data)
  
  reward[i, 1:n_trials] <- subj_data$reward
  choice[i, 1:n_trials] <- ifelse(subj_data$choice == 1 | subj_data$choice == 0, subj_data$choice, -1)
  choice_two[i, 1:n_trials] <- ifelse(subj_data$choice == 1, 1, ifelse(subj_data$choice == 2, 0, -1))
}

# Check if there are any NA values in the matrices (should be no NAs if placeholders are used)
print(any(is.na(reward)))  # Should be FALSE if all values are valid
print(any(is.na(choice)))  # Should be FALSE if all values are valid
print(any(is.na(choice_two)))  # Should be FALSE if all values are valid

stan_data <- list(
  NS = NS,  # Number of subjects
  MT = MT,  # Maximum number of trials
  NC = NC,  # Number of choices
  K = K,  # Number of coefficients for GLM
  Tsubj = trial_counts$n_trials,  # Number of trials per subject
  NT = trial_counts$n_trials,  # Number of trials per subject (NT and Tsubj should be the same)
  rew = reward,  # Reward
  choice = choice,  # Choice
  choice_two = choice_two  # Choice two
)


# Check the structure of the prepared data
str(stan_data)


mod <- cmdstan_model("my_hierarchical_6.stan")

fit <- mod$sample(
  data = stan_data,
  parallel_chains = 4
)

# fit <- mod$sample(
#   data = stan_data,
#   seed = 123,
#   chains = 4,
#   parallel_chains = 4,
#   iter_warmup = 2000,
#   iter_sampling = 2000,
#   adapt_delta = 0.99,
#   max_treedepth = 15
# )

fit$summary("alpha")

mcmc_hist(fit$draws("alpha"), binwidth = 0.025)
mcmc_hist(fit$draws("beta"), binwidth = 0.025)


fit$summary("b_mean")




###############################################################
# Extract draws
draws <- fit$draws()

library(posterior)

# Summarize the parameters
summary_fit <- summary(draws)

# Display the summary for specific parameters
print(summary_fit)

# Extract specific parameters
alpha_draws <- as_draws_matrix(draws, variable = "alpha")
b_mean_draws <- as_draws_matrix(draws, variable = "b_mean")
b_sd_draws <- as_draws_matrix(draws, variable = "b_sd")

# Summarize alpha
alpha_summary <- summarize_draws(alpha_draws)
print(alpha_summary)

# Summarize b_mean
b_mean_summary <- summarize_draws(b_mean_draws)
b_mean_summary |> as.data.frame() 

# Summarize b_sd
b_sd_summary <- summarize_draws(b_sd_draws)
print(b_sd_summary)

# Summarize the natural scale for b_mean
b_mean_natural_scale <- exp(b_mean_draws)
b_mean_natural_summary <- summarize_draws(b_mean_natural_scale)
print(b_mean_natural_summary)

# Summarize the natural scale for b_sd
b_sd_natural_scale <- exp(b_sd_draws)
b_sd_natural_summary <- summarize_draws(b_sd_natural_scale)
print(b_sd_natural_summary)



# Extract and summarize alpha parameters for each subject
alpha_draws <- as_draws_matrix(draws, variable = "alpha")
alpha_summary <- summarize_draws(alpha_draws)





# Extract draws
draws <- fit$draws()

# Convert draws to a data frame for easier manipulation
draws_df <- as_draws_df(draws)

# Filter only alpha and beta parameters
alpha_draws <- draws_df %>% select(starts_with("alpha"))
beta_draws <- draws_df %>% select(starts_with("beta"))

# Summarize alpha parameters
alpha_summary <- summarize_draws(alpha_draws)

# Summarize beta parameters
beta_summary <- summarize_draws(beta_draws)

# Print the summaries
print("Alpha parameters for each subject:")
print(alpha_summary)

print("Beta parameters for each subject:")
print(beta_summary)

# ------------------------------------------------------------------------------

mod <- cmdstan_model("new_syntax_Block4.stan")








