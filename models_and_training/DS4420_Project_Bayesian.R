# Load features from our CNN model in Python
train_data <- read.csv('/Users/colinchu/Documents/ds4420/train_features.csv')
test_data <- read.csv('/Users/colinchu/Documents/ds4420/test_features.csv')

# Split data to training and test
X_train <- as.matrix(cbind(1, train_data[,-ncol(train_data)]))
X_test <- as.matrix(cbind(1, test_data[,-ncol(test_data)]))
y_train <- train_data$label
y_test <- test_data$label

# Normal model with unknown variance
n_classes <- 5
n_features <- ncol(X_train)
no_samples <- 2000

prior_w <- matrix(0, nrow = n_features, ncol = 1)
prior_Sigma <- diag(n_features) * 10

#since we have 5 superclasses, we need an array for all w samples
all_w_samples <- list()

# Sigmoid function
sigmoid <- function(w, X) {
  1 / (1 + exp(-X %*% w))
}

# Log prior
log_prior <- function(w, prior_Sigma) {
  -0.5 * t(w) %*% solve(prior_Sigma) %*% w
}

# Log likelihood
log_likelihood <- function(w, X, y) {
  p <- sigmoid(w, X)
  sum(y * log(p + 1e-10) + (1 - y) * log(1 - p + 1e-10)) # add 1e-10 to avoid log(0)
}

# Log joint (prior + likelihood)
log_joint <- function(w, X, y, prior_Sigma) {
  log_likelihood(w, X, y) + log_prior(w, prior_Sigma)
}

prior_w <- matrix(0, nrow = n_features, ncol = 1)
prior_Sigma <- diag(n_features) * 10

# Metropolis sampler
metropolis_bayesian_logistic <- function(X, y_binary, n_features,
                                n = 2000, sigma = 1, burn = 0.25) {
  
  total_samples <- as.integer(n / (1 - burn))
  m <- as.integer(total_samples * burn)  # burn-in amount
  accepted <- 0
  
  # Pre-allocate (same as course file)
  samples <- matrix(0, nrow = total_samples, ncol = n_features)
  w_tilde <- matrix(0, nrow = n_features, ncol = 1)
  
  for (i in 1:total_samples) {
    w_proposal <- w_tilde + matrix(rnorm(n_features, 0, sigma), ncol = 1)
    
    # Acceptance ratio (log scale for numerical stability)
    log_alpha <- log_joint(w_proposal, X, y_binary, prior_Sigma) -
      log_joint(w_tilde, X, y_binary, prior_Sigma)
    
    if (log(runif(1)) < log_alpha) {
      w_tilde <- w_proposal
      accepted <- accepted + 1
    }
    
    # Store sample (same as course file)
    samples[i, ] <- w_tilde
  }
  
  # Acceptance rate (ideally around 23.4%)
  acceptance_rate <- accepted / total_samples
  cat(sprintf("Acceptance rate: %.3f\n", acceptance_rate))
  
  # Discard burn-in (same as course file)
  samples <- samples[(m + 1):nrow(samples), ]
  
  return(samples)
}

class_names <- c("Store", "Home", "Public", "Leisure", "Working")

# Outer loop: one binary classifier per super-category
for (class_id in 0:(n_classes - 1)) {
  cat(sprintf("Fitting class %d/5: %s\n", class_id + 1, class_names[class_id + 1]))
  
  # Binary labels: 1 if this super-category, 0 otherwise
  y <- as.numeric(y_train == class_id)
  
  all_w_samples[[class_id + 1]] <- metropolis_bayesian_logistic(X_train, y, n_features)
}

for (class_id in 1:n_classes) {
  plot(all_w_samples[[class_id]][, 1], type = "l",
       main = sprintf("Trace - %s", class_names[class_id]),
       ylab = "w", xlab = "iteration")
}

for (class_id in 1:n_classes) {
  acf(all_w_samples[[class_id]][, 1],
      main = sprintf("ACF - %s", class_names[class_id]))
}

# Posterior mean weights
posterior_w <- do.call(rbind, lapply(all_w_samples, colMeans))

# Predict probabilities for test set
probs <- matrix(0, nrow = nrow(X_test), ncol = n_classes)
for (class_id in 1:n_classes) {
  w <- matrix(posterior_w[class_id, ], ncol = 1)
  probs[, class_id] <- sigmoid(w, X_test)
}

# Per class accuracies
y_pred <- apply(probs, 1, which.max) - 1
accuracy <- mean(y_pred == y_test)
cat(sprintf("Test accuracy: %.4f\n", accuracy))

# Per class accuracy
for (class_id in 0:(n_classes - 1)) {
  class_acc <- mean(y_pred[y_test == class_id] == class_id)
  cat(sprintf("%s accuracy: %.4f\n", class_names[class_id + 1], class_acc))
}