library(tidyverse)

set.seed(123)  # For reproducibility


simulate_bandit <- function(n_subjects, max_trials, alphas, betas) {
  # Initialize data frame to store results
  data <- data.frame(subjID = integer(), choice = integer(), outcome = integer())
  
  for (subj in 1:n_subjects) {
    Q <- c(0, 0)  # Initialize Q-values for the two options
    alpha <- alphas[subj]
    beta <- betas[subj]
    
    # Randomly determine the number of trials for this subject
    n_trials <- sample(1:max_trials, 1)
    
    for (trial in 1:n_trials) {
      # Compute choice probabilities using softmax function
      prob <- exp(beta * Q) / sum(exp(beta * Q))
      
      # Make a choice based on the probabilities
      choice <- sample(1:2, 1, prob = prob)
      
      # Simulate an outcome: let's assume each option has a 50% chance of reward
      outcome <- ifelse(runif(1) < 0.5, 1, -1)
      
      # Store the result
      data <- rbind(data, data.frame(subjID = subj, choice = choice, outcome = outcome))
      
      # Update the Q-value based on the Rescorla-Wagner rule
      Q[choice] <- Q[choice] + alpha * (outcome - Q[choice])
    }
  }
  
  return(data)
}

# Set parameters for each subject
n_subjects <- 5
max_trials <- 300
alphas <- runif(n_subjects, 0.05, 0.15)  # Random alphas between 0.05 and 0.15
betas <- runif(n_subjects, 4, 6)         # Random betas between 4 and 6

# Simulate data
simulated_data <- simulate_bandit(
  n_subjects = n_subjects, max_trials = max_trials, 
  alphas = alphas, betas = betas)

# Check the first few rows of the simulated data
head(simulated_data)

dim(simulated_data)

simulated_data |> 
  group_by(subjID) |> 
  summarize(
    n = n()
  )

write.table(
  simulated_data, 
  "simulated_bandit_data.txt", sep = "\t", row.names = FALSE, col.names = TRUE
)

model_fit <- bandit2arm_delta(
  data = "simulated_bandit_data.txt",
  niter = 4000,
  nwarmup = 1000,
  nchain = 4,
  ncore = 8
)

plot(model_fit, type = "trace")
rhat(model_fit)
plot(model_fit)
printFit(model_fit)

ind_pars <- model_fit$allIndPars
print(ind_pars)

# The function bandit2arm_delta() recovers adequately the alpha and beta
# parameters, for multiple subjects with different number of trials each.

# ------------------------------------------------------------------------------
# Real data from the Groundhog-Day project.

#' subjID: A unique identifier for each subject in the data-set.
#' choice: Integer value representing the option chosen on the given 
#'         trial: 1 or 2.
#' outcome: Integer value representing the outcome of the given trial 
#'          (where reward == 1, and loss == -1).

real_data <- rio::import("groundhog_hddmrl_data.csv")

d <- data.frame(
  subjID = as.numeric(as.factor(as.character(real_data$user_id))),
  choice = ifelse(
    real_data$response == 1, 1, ifelse(real_data$response == 0, 2, NA)
  ),
  outcome = ifelse(
    real_data$feedback == 1, 1, ifelse(real_data$feedback == 0, -1, NA)
  )
)

d_small <- d |> 
  dplyr::filter((subjID > 10) & (subjID < 21))
dim(d_small)

real_data |> 
  group_by(ema_number) |> 
  summarize(
    y = mean(response),
    n = n()
  )

dd <- real_data |> 
  dplyr::filter(ema_number < 11)

dd$is_reversal <- ifelse(dd$trial_in_block < 16, 1, 0)

fm <- glmer(
  response ~ ema_number + mood_pre + is_reversal + 
    (mood_pre + ema_number + is_reversal | user_id),
  family = binomial(),
  data = dd
)


write.table(
  d_small, 
  "groundhog_data.txt", sep = "\t", row.names = FALSE, col.names = TRUE
)

fit <- bandit2arm_delta(
  data = "groundhog_data.txt",
  niter = 4000,
  nwarmup = 1000,
  nchain = 4,
  ncore = 8,
  vb = TRUE
)

plot(fit, type = "trace")
rhat(fit)
plot(fit)
printFit(fit)

ind_pars <- fit$allIndPars
print(ind_pars)


