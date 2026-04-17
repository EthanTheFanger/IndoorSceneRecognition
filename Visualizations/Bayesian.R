library(mvtnorm)

# Load features from our CNN model in Python
train_data <- read.csv("C:/Users/colin/OneDrive/Documents/ds4420/train_features_normalized.csv")
test_data <- read.csv("C:/Users/colin/OneDrive/Documents/ds4420/test_features_normalized.csv")

# Split data to training and test
X_train <- as.matrix(cbind(1, train_data[,-ncol(train_data)]))
X_test <- as.matrix(cbind(1, test_data[,-ncol(test_data)]))
y_train <- train_data$label
y_test <- test_data$label

n_classes <- 5
n_features <- ncol(X_train)
no_samples <- 10000

# Sigmoid function
sigmoid <- function(w, X) {
  1 / (1 + exp(-X %*% w))
}

# Define prior, likelihood and joint
mu0 = array(0, n_features)
Sigma0 <- diag(n_features) * 1


# Without log likelihood/priors, we run into underflowing issues (probabilities multiply to 0)
# So we use a log likelihood/prior, and add them instead of multiplying
likelihood <- function(w, X, y) {
  p <- sigmoid(w, X)
  sum(y * log(p + 1e-10) + (1 - y) * log(1 - p + 1e-10)) #add 1e-10 to prevent log(0)
}

prior <- function(w) { 
  log(dmvnorm(as.vector(w), mean = mu0, sigma = Sigma0))
}

p <- function(w, X, y) {
  likelihood(w, X, y) + prior(w)
}

#since we have 5 superclasses, we need an array for all w samples
all_w_samples <- list()

# Metropolis sampler
metropolis_bayesian_logistic <- function(X, y, n_features,
                                n = no_samples, sigma = 0.05, burn = 0.25) {
  
  total_samples <- as.integer(n / (1 - burn))
  m <- as.integer(total_samples * burn)  # burn-in amount
  accepted <- 0
  
  samples <- matrix(0, nrow = total_samples, ncol = n_features)
  w_tilde <- matrix(0, nrow = n_features, ncol = 1)
  
  for (i in 1:total_samples) {
    w_proposal <- w_tilde + matrix(rnorm(n_features, 0, sigma), ncol = 1)
    
    # Acceptance ratio (subtract logs since we're using a log joint)
    alpha <- p(w_proposal, X, y) - p(w_tilde, X, y)
    
    if (log(runif(1)) < alpha) {
      w_tilde <- w_proposal
      accepted <- accepted + 1
    }
    
    # Store sample
    samples[i, ] <- w_tilde
  }
  
  # Acceptance rate (ideally around 23.4%)
  acceptance_rate <- accepted / total_samples
  cat(sprintf("Acceptance rate: %.3f\n", acceptance_rate))
  
  # Discard burn-in
  samples <- samples[(m + 1):nrow(samples), ]
  
  return(samples)
}

class_names <- c("Store", "Home", "Public", "Leisure", "Working")

# Outer loop: one binary classifier per super-category
for (class_id in 0:(n_classes - 1)) {
  cat(sprintf("Fitting class %d/5: %s\n", class_id + 1, class_names[class_id + 1]))
  
  y <- as.numeric(y_train == class_id)
  
  all_w_samples[[class_id + 1]] <- metropolis_bayesian_logistic(X_train, y, n_features)
}

# Thinning (keep every 25th sample)
for (class_id in 1:n_classes) {
  all_w_samples[[class_id]] <- all_w_samples[[class_id]][
    seq(1, nrow(all_w_samples[[class_id]]), by = 25), ]
}

# Trace plots
par(mfrow = c(2, 3))
for (class_id in 1:n_classes) {
  plot(all_w_samples[[class_id]][, 1], type = "l",
       main = sprintf("Trace - %s", class_names[class_id]),
       ylab = "w", xlab = "iteration")
}

# ACF plots
par(mfrow = c(2, 3))
for (class_id in 1:n_classes) {
  acf(all_w_samples[[class_id]][, 1],
      main = sprintf("ACF - %s", class_names[class_id]))
}

# Model accuracy
posterior_w <- do.call(rbind, lapply(all_w_samples, colMeans))
probs <- matrix(0, nrow = nrow(X_test), ncol = n_classes)
for (class_id in 1:n_classes) {
  w <- matrix(posterior_w[class_id, ], ncol = 1)
  probs[, class_id] <- sigmoid(w, X_test)
}

y_pred <- apply(probs, 1, which.max) - 1
accuracy <- mean(y_pred == y_test)
cat(sprintf("Test accuracy: %.4f\n", accuracy))

# Confusion Matrix
library(ggplot2)
library(reshape2)
par(mfrow = c(1, 1))
conf_matrix <- table(Predicted = y_pred, Actual = y_test)

# Plot confusion matrix
conf_df <- melt(conf_matrix)
conf_df$Actual <- factor(class_names[conf_df$Actual + 1], levels = class_names)
conf_df$Predicted <- factor(class_names[conf_df$Predicted + 1], levels = rev(class_names))
ggplot(conf_df, aes(x = Actual, y = Predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  scale_fill_distiller(palette = "Blues", direction = 1) +
  labs(title = "Bayesian Model - Confusion Matrix") + 
  theme_minimal()

# Normalized version
conf_matrix_norm <- prop.table(conf_matrix, margin = 2)
conf_norm_df <- melt(conf_matrix_norm)
conf_norm_df$Actual <- factor(class_names[conf_norm_df$Actual + 1], levels = class_names)
conf_norm_df$Predicted <- factor(class_names[conf_norm_df$Predicted + 1], levels = rev(class_names))

ggplot(conf_norm_df, aes(x = Actual, y = Predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  scale_fill_distiller(palette = "Blues", direction = 1) +
  labs(title = "Bayesian Model - Confusion Matrix (Normalized)") +
  theme_minimal()