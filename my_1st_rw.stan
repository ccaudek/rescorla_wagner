data {
    int<lower=1> nTrials; // Number of trials
    array[nTrials] int choice; // Choices are 1 or 2, hence int type
    array[nTrials] real reward; // Rewards, since they can be -1 or 1
} 

parameters {
    real<lower=0, upper=1> alpha; // Learning rate
    real<lower=0, upper=20> tau; // Softmax inverse temperature
}

model {
    // Priors
    alpha ~ beta(2, 2); // Prior for alpha
    tau ~ gamma(2, 0.1); // Prior for tau
    
    vector[2] v = [0, 0]'; // Initial values for the two choices
    vector[2] p; // Probability vector for the choices
    real pe; // Prediction error
    
    for (t in 1:nTrials) {
        p = softmax(tau * v); // Compute action probabilities via softmax
        choice[t] ~ categorical(p); // Sampling statement for choice
        
        pe = reward[t] - v[choice[t]]; // Compute prediction error for chosen value
        v[choice[t]] = v[choice[t]] + alpha * pe; // Update the value of the chosen option
    }
}


