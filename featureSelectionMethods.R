# Function：pearsonCorrelation 
# Purpose: Performs feature selection using the Pearson correlation coefficient to remove highly correlated features and evaluates the resulting feature subset with a classifier.
# Parameters:
#  - dataset: The dataset to be processed, where the last column is the target variable.
#  - seed: The seed value used for data splitting.
#  - classifier: A function that evaluates the feature subset and returns classification result.
#  - Name: The name of the classifier, used in the result output.
#  - cutoff: A threshold for identifying correlated features. Features with correlation above this value are removed.
#  - datasetName: The name of the dataset, used in the result output.
#Returns:
#A list summarizing the dataset name, classifier, correlation cutoff, number of selected features, and the resulting accuracy.
pearsonCorrelation <- function(dataset,seed,classifier,Name,cutoff,datasetName){
  # Initialize a data frame with column names representing each feature in the dataset, 
  # excluding the last column (target variable).
  fullindex <- data.frame(
    matrix(data = 1,
           ncol = ncol(dataset)-1,
           nrow = 1,
           dimnames = list(c("whetherSelect"),
                           c(colnames(dataset[,-ncol(dataset)]))))
    
  )
  
  # Process the data using the provided seed and the initialized fullindex.
  data <- processing_data(dataset,seed,fullindex)
  
  # Calculate the Pearson correlation coefficient for each pair of features (excluding the target variable).
  corCoef <- cor(data$train_set[,-ncol(data$train_set)],method = "pearson")
  
  # Use the findCorrelation function to identify correlated features, based on the given cutoff value.
  corFeat <- findCorrelation(corCoef, cutoff = cutoff)
  
  # Initialize an index data frame to keep track of whether a feature is selected (1) or not (0).
  index <- data.frame(
    matrix(data = 1,
           ncol = ncol(data$train_set)-1)
  )
  
  # Set the index values for highly correlated features to 0 (they are excluded).
  for ( i in corFeat){
    index[i] = 0
  }
  
  # Evaluate the classifier's performance with the selected features and calculate the error.
  error = classifier(dataset,seed,index,returnType = "error")
  
  # Return a list containing the dataset name, classifier name, cutoff, number of selected features, and the accuracy (1 - error).
  result <- list(
    dataset = datasetName,
    classifierName = Name,
    cutoff = cutoff,
    numOfSelectedFea = sum(index),
    accuracy = 1 - error
  )
}

# Function: mutualInfo
# Purpose: This function performs feature selection using Mutual Information (MI) to evaluate the relationship between features 
#          and the target variable. It selects features with mutual information above a given threshold and evaluates a classifier's performance.
# Parameters:
#   - dataset: The dataset to process, where the last column is assumed to be the target variable.
#   - seed: The seed value used for data splitting.
#   - classifier: A function that evaluates the feature subset and returns classification result.
#   - Name: The name of the classifier, used in the result output.
#   - boundary: The mutual information threshold used to select features.
#   - datasetName: The name of the dataset, used in the result output.
#Returns:
#A list summarizing the dataset name, classifier, boundary, number of selected features, and the resulting accuracy.
mutualInfo <- function (dataset,seed,classifier,Name,boundary,datasetName){
  #Initialize a data frame with column names representing each feature in the dataset, excluding the last column (target variable).
  fullindex <- data.frame(
    matrix(data = 1,
           ncol = ncol(dataset)-1,
           nrow = 1,
           dimnames = list(c("whetherSelect"),
                           c(colnames(dataset[,-ncol(dataset)]))))
    
  )
  # Process the data using the provided seed and the initialized fullindex.
  data <- processing_data(dataset,seed,fullindex)
  
  # Extract the feature columns from the training set, excluding the target variable.
  data_fea <- data$train_set [,-ncol(data$train_set)]
  
  # Discretize the features using the 'arules::discretize' function to convert them into categorical values based on intervals.
  data_category = data.frame(lapply(data_fea, function(x) arules::discretize(x, method = "interval", breaks  = length(unique(data$train_set$device_category)))))
  
  # Add the target variable (device_category) to the discretized features.
  data_category$device_category = data$train_set$device_category
  
  # Calculate the mutual information between each feature and the target variable.
  miD <- apply(data_category[ ,-ncol(data_category)], 2, function(x) mutinformation(x,data_category[,ncol(data_category)]))
  
  # Sort the mutual information values in descending order.
  miD <- sort(miD, decreasing = TRUE)
  
  # Determine the number of selected features based on the mutual information threshold (boundary).
  numOfSelectedFea = sum(miD>=boundary)
  
  # Select the top features based on the mutual information values.
  top_features <- names(miD[1:numOfSelectedFea])
  
  # Initialize an index data frame to track whether a feature is selected (1) or not (0).
  index <- data.frame(
    matrix(data = 0,
           ncol = ncol(data$train_set)-1,
           nrow = 1,
           dimnames = list(c("whetherSelect"),
                           c(colnames(data$train_set[,-ncol(data$train_set)]))))
    
  )
  
  # Set the index values for the top features to 1 (they are selected).
  for (i in top_features){
    index[[i]] = 1
  }
  
  # Evaluate the classifier's performance with the selected features and calculate the error.
  error = classifier(dataset,seed,index,returnType = "error")
  
  # Return a list containing the dataset name, classifier name, mutual information threshold, 
  # number of selected features, and the accuracy (1 - error).
  result <- list(
    dataset = datasetName,
    classifierName = Name,
    boundary = boundary,
    numOfSelectedFea = numOfSelectedFea,
    accuracy = 1-error
  )
}

# Function: BGWO
# Purpose: This function implements the Binary Grey Wolf Optimizer (BGWO) for feature selection.
#          It uses the grey wolf optimization algorithm to find the best feature subset based on a fitness function.
# Parameters:
#   - data: The dataset to process, where the last column is assumed to be the target variable.
#   - seed: The seed value used for data splitting.
#   - N_wolf: The number of wolves
#   - max_Iter: The maximum number of iterations for the optimization process.
#   - fitFunctionName: The name of the fitness function used to evaluate the quality of feature subsets.
#   - algoName: The name of the classifier, used in the result output.
#   - perErr: The weight assigned to the error rate when calculating the fitness value
# Returns:
#   - bestResult: A list containing:
#       - optimizer: The optimizer used for feature selection.
#       - algorithm: The classifier used.
#       - numAlphaFeature: The number of features selected in the best solution.
#       - smallestError: The smallest error achieved.
#       - alphaFeatSetIndex: The binary index of the selected features.
#Binary Grey Wolf Optimizer(BGWO) implementation
BGWO <- function (data,seed,N_wolf,max_Iter,fitFunctionName,algoName,perErr){
  # Get the number of features (columns minus the target variable)
  dime = ncol(data)-1
  
  # Initialize wolves' positions (feature indices)
  featIndex <- matrix(data = 0,nrow = N_wolf,ncol = dime,dimnames = list(c(1:N_wolf),c(1:dime)))
  for (i in 1:N_wolf) {
    for (j in 1:dime) {
      if(i >= 2){
        if (runif(1,min=0,max=1) > 0.5)
          featIndex[i,j] <- 1
      }# Leader starts with all features selected
      if(i == 1){
        featIndex[i,j] <-1
      }
    }
  }
  
  # Initialize fitness values for each wolf
  fit <- matrix(
    data = 0,
    nrow = 1,
    ncol = N_wolf,
    dimnames = list(c(1),
                    c(1:N_wolf))
  )
  for (i in 1:N_wolf) {
    fit[i] = fitFunctionName(data,seed,featIndex[i,],perErr,"fitnessValue")
  }
  
  # Sort wolves by fitness and identify Alpha, Beta, Delta wolves
  fit <- sort(fit,index.return = TRUE)
  fitAlpha = fit$x[1]
  featAlpha = featIndex[fit$ix[1],]
  fitBeta = fit$x[2]
  featBeta = featIndex[fit$ix[2],]
  fitDelta = fit$x[3]
  featDelta = featIndex[fit$ix[3],]
  
  # Initialize error tracking
  error <- data.frame(
    ilteration = c(1:100),
    err = rep(c(0),100)
  )
  
  # Initialize progress bar
  pb <- txtProgressBar(min = 0, max = max_Iter, style = 3, width = 50, char = "=")
  
  # Main optimization loop
  t <- 1
  while (t <= max_Iter){
    a = 2 - 2*(t / max_Iter)  # Decrease coefficient linearly
    for (i in 1 : N_wolf){
      for (d in 1 : dime){
        # Compute distances and update positions using Alpha, Beta, Delta wolves
        C1 = 2 * runif(1,min=0,max=1)
        C2 = 2 * runif(1,min=0,max=1)
        C3 = 2 * runif(1,min=0,max=1)
        Dalpha = abs(C1 * featAlpha[d] - featIndex[i,d])
        Dbeta  = abs(C2 * featBeta[d]  - featIndex[i,d])
        Ddelta = abs(C3 * featDelta[d] - featIndex[i,d])
        A1 = 2 * a * runif(1,min=0,max=1) - a
        A2 = 2 * a * runif(1,min=0,max=1) - a
        A3 = 2 * a * runif(1,min=0,max=1) - a
        feat1 = featAlpha[d] - A1 * Dalpha
        feat2 = featBeta[d]  - A2 * Dbeta
        feat3 = featDelta[d] - A3 * Ddelta
        featn = (feat1 + feat2 + feat3) / 3
        TF = 1 / (1 + exp(-10 * (featn - 0.5)))  # Sigmoid function for feature selection
        if (TF >= runif(1,min=0,max=1)){
          featIndex[i,d] = 1
        }else{
          featIndex[i,d] = 0
        }
      }
    }
    # Update fitness and identify new Alpha, Beta, Delta wolves
    for (i in 1:N_wolf){
      fit[i] = fitFunctionName(data,seed,featIndex[i,],perErr,"fitnessValue")
      if (as.data.frame(fit[i]) < fitAlpha
          |(as.data.frame(fit[i]) == fitAlpha 
            & sum(featIndex[i,])<sum(featAlpha))){
        fitAlpha = fit[i]
        featAlpha = featIndex[i,]
      }
      if ((as.data.frame(fit[i])<fitBeta 
           & as.data.frame(fit[i])>fitAlpha)
          |(as.data.frame(fit[i]) == fitBeta 
            & sum(featIndex[i,])<sum(featBeta)) ){
        fitBeta = fit[i]
        featBeta = featIndex[i,]
      }
      if ((as.data.frame(fit[i]) < fitDelta 
           & as.data.frame(fit[i])>fitAlpha 
           & as.data.frame(fit[i])>fitBeta)
          |(as.data.frame(fit[i]) == fitDelta 
            & sum(featIndex[i,])<sum(featDelta))){
        fitDelta = fit[i]
        featDelta = featIndex[i,]
      }
    }
    # Track error and update progress bar
    error$err[t] <- fitAlpha
    setTxtProgressBar(pb, t) 
    t = t + 1
  }
  close(pb) # Close progress bar
  beep(2)   # Play notification sound
  
  # Return the best feature set and summary
  bestResult <- feature_set_summary(perErr, "BGWO", fitAlpha, dime, featAlpha, algoName)
  return(bestResult)
}

# Function:BGA
# Purpose: This function implements a Binary Genetic Algorithm (BGA) for feature selection. The goal is to find the best subset of features
# from the dataset that optimizes a custom fitness function. The BGA works by evolving a population of feature subsets through
# crossover and mutation over multiple generations, guided by a fitness function that evaluates the quality of the feature subsets.
# Parameters:
#   - data: A data frame containing the dataset with features and target variable.
#   - seed: The seed value used for data splitting.
#   - population_size: The number of individuals in the population for the genetic algorithm.
#   - max_Iter: The maximum number of iterations (generations) for the algorithm to run.
#   - fitFunctionName: A custom function to evaluate the fitness of a feature subset.
#   - algoName: The name of the algorithm, used for output purposes.
#   - perErr: The weight assigned to the error rate when calculating the fitness value
#   - crossover: The probability of performing crossover between individuals during evolution.
#   - mutation: The probability of performing mutation on individuals during evolution.
# Returns:
#   - bestResult: A list containing:
#       - optimizer: The optimizer used for feature selection.
#       - algorithm: The classifier used.
#       - numAlphaFeature: The number of features selected in the best solution.
#       - smallestError: The smallest error achieved.
#       - alphaFeatSetIndex: The binary index of the selected features.
# Binary Genetic Algorithm (BGA) implementation
BGA <- function(data, seed, population_size, max_Iter, fitFunctionName, algoName, perErr, crossover, mutation) {
  
  # Function to generate the initial population for the Genetic Algorithm
  generate_initial_population <- function(object) {
    population_size <- object@popSize # Number of individuals in the population
    chromosome_length <- object@nBits # Length of each chromosome (number of features)
    # Randomly generate binary population matrix
    population <- matrix(sample(0:1, population_size * chromosome_length, replace = TRUE), nrow = population_size, ncol = chromosome_length)
    population[1, ] <- rep(1, chromosome_length) # First individual selects all features
    return(population)
  }
  
  # Function to evaluate the fitness of a feature subset
  evaluate_feature_subset <- function(feature_subset,fitFunctionName,data, seed_, a) {
    Index <- as.numeric(feature_subset) # Convert feature subset to numeric
    if (sum(Index) == 0) { # If no features are selected, return fitness 0
      return(0)
    }
    seed = seed_ 
    fitnessValue <- fitFunctionName(data,seed,Index,a,"fitnessValue") # Calculate fitness
    return(1- fitnessValue)
  }
  
  dime = ncol(data)-1 # Number of features
  # Run the Genetic Algorithm
  result <- ga(type = "binary",                 # Specify binary encoding
               fitness = evaluate_feature_subset, # Fitness function to evaluate subsets
               fitFunctionName = fitFunctionName, # Custom fitness function
               data = data,                     # Dataset
               seed_ = seed,                    # Seed for split dataset
               a = perErr,                      # The weight assigned to the error rate when calculating the fitness value
               population = generate_initial_population, # Initial population generator
               popSize = population_size,       # Population size
               nBits = dime,                    # Chromosome length (number of features)
               maxiter = max_Iter,              # Maximum number of iterations
               pcrossover = crossover,          # Crossover probability
               pmutation = mutation,            # Mutation probability
               run = 20                         # Max generations without improvement
  )
  
  # Generate summary of the best solution
  bestResult <- feature_set_summary(perErr, "BGA", 1-result@fitnessValue,dime , result@solution[1,], algoName)
  return(bestResult) # Return the best result
}

# Function: BPSO
# Purpose: This function implements the Binary Particle Swarm Optimization (BPSO) algorithm to perform feature selection.
# The algorithm updates particle positions based on their personal best positions and the global best position in the swarm. 
# The optimization continues until the maximum iterations 
# are reached or no improvement occurs for a specified number of consecutive iterations.
#
# Parameters:
#   - data: The dataset used for feature selection (excluding the target variable).
#   - seed: The seed value used for data splitting to ensure reproducibility.
#   - n_particles: The number of particles (individual feature subsets) in the swarm.
#   - max_Iter: The maximum number of iterations for the optimization.
#   - fitFunctionName: The fitness function used to evaluate feature subsets.
#   - algoName: The name of the classifier.
#   - perErr: The weight assigned to the error rate when calculating the fitness value.
#   - inertia: The inertia coefficient that controls the impact of the previous velocity on the current velocity.
#   - c1: The cognitive coefficient that controls the influence of the particle's own best position.
#   - c2: The social coefficient that controls the influence of the global best position.
#   - Vmax: The maximum allowed velocity for the particles' movement.
# Returns:
#   - bestResult: A list containing:
#       - optimizer: The optimizer used for feature selection.
#       - algorithm: The classifier used.
#       - numAlphaFeature: The number of features selected in the best solution.
#       - smallestError: The smallest error achieved.
#       - alphaFeatSetIndex: The binary index of the selected features.
# Binary Particle Swarm Optimization Algorithm (BPSO) implementation
BPSO <- function(data, seed, n_particles, max_Iter, fitFunctionName, algoName, perErr, inertia, c1, c2, Vmax) {
  # Sigmoid function for mapping velocities to probabilities
  sigmoid <- function(x) {
    return(1 / (1 + exp(-x)))
  }
  
  # Initialize particle positions (binary feature subsets)
  dime <- ncol(data) - 1  # Number of features
  first_particle <- matrix(rep(1, dime), nrow = 1, ncol = dime) # First particle selects all features
  # Random initialization for others
  other_particles <- matrix(sample(c(0, 1), (n_particles - 1) * dime, replace = TRUE), nrow = n_particles - 1, ncol = dime)
  particles <- rbind(first_particle, other_particles) # Combine all particles
  
  # Create a record to store particles and their fitness values
  particles_record <- data.frame(cbind(particles, rep(0, n_particles)))
  colnames(particles_record) <- c(colnames(data)[1:dime], 'value') 
  
  # Compute initial fitness values for all particles
  for (i in 1:n_particles) {
    particles_record$value[i] <- fitFunctionName(data, seed, particles_record[i, 1:dime], perErr,"fitnessValue")
  }
  
  # Initialize velocities as a zero matrix
  velocities <- as.data.frame(matrix(0, nrow = n_particles, ncol = dime)) 
  
  # Record individual best positions and their fitness values
  individual_best_record <- particles_record
  colnames(individual_best_record) <- c(colnames(data)[1:dime], 'best_value') 
  
  # Identify the global best position and fitness value
  global_best_value <- min(individual_best_record$best_value)
  global_best_position <- individual_best_record[which.min(individual_best_record$best_value), 1:dime]
  
  # Initialize progress bar
  pb <- txtProgressBar(min = 0, max = max_Iter, style = 3, width = 50, char = "=")
  
  # Optimization loop
  repeat_best = 0 # Track consecutive iterations with no improvement
  for (iter in 1:max_Iter) {
    # Random coefficients for velocity updates
    r1 <- matrix(runif(n_particles * dime), nrow = n_particles, ncol = dime)
    r2 <- matrix(runif(n_particles * dime), nrow = n_particles, ncol = dime)
    
    # Update velocities using PSO formula
    velocities <- inertia * velocities +
      c1 * r1 * (as.matrix(individual_best_record[, 1:dime]) - as.matrix(particles_record[, 1:dime])) +
      c2 * r2 * (matrix(rep(as.numeric(global_best_position), each = n_particles), nrow = n_particles, byrow = TRUE) - as.matrix(particles_record[, 1:dime]))
    
    # Cap velocities to the maximum allowed value (Vmax)
    velocities <- pmin(velocities, Vmax)
    
    # Update particle positions using sigmoid transformation and binary thresholding
    particles_record[, 1:dime] <- ifelse(runif(n_particles * dime) < sigmoid(velocities), 1, 0)
    
    # Recalculate fitness values for the updated particles
    particles_record$value <- apply(particles_record[, 1:dime], 1, function(x) fitFunctionName(data, seed, x, perErr,"fitnessValue"))
    
    # Update individual best positions and values
    individual_best_record[particles_record$value < individual_best_record$best_value, ] <- particles_record[particles_record$value < individual_best_record$best_value, ]
    
    # Update global best position and fitness value
    min_value_index <- which.min(particles_record$value)
    if (particles_record$value[min_value_index] < global_best_value) {
      global_best_position <- particles_record[min_value_index, 1:dime]
      global_best_value <- particles_record$value[min_value_index]
      repeat_best = 0 # Reset counter for consecutive stagnant iterations
    }
    
    # Update progress bar
    setTxtProgressBar(pb, iter) 
    
    # Stop if no improvement for 20 consecutive iterations
    repeat_best = repeat_best + 1
    if (repeat_best >= 20){
      break
    }
  }
  
  # Close progress bar and notify user
  close(pb)
  beep(2)
  
  # Summarize and return the best feature subset and its fitness
  bestResult <- feature_set_summary(perErr, "BPSO", global_best_value, dime, global_best_position, algoName)
  return(bestResult)
}






