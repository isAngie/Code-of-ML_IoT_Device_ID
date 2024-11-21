# This file contains a collection of fitness functions implemented for various machine learning models, 
# including SVM, Neural Networks, Decision Trees, and Random Forests. 
# Each function is designed to evaluate the performance of models on a specific dataset. 
# 
# Parameter descriptions for the fitness functions:
# - `data`: The dataset containing features and the target variable (`device_category`), used for training and testing.
# - `seed`: The seed value passed to the `processing_data` function to split the dataset into training (70%) 
#   and testing (30%) subsets, ensuring reproducibility of the data split.
# - `Index`: The index of the selected feature set, provided by feature selection methods, 
#   to determine which features are used in training and testing.
# - `perErr`: The weight of the error rate in the calculation of the fitness value, balancing the impact of errors in wrapper-based feature selection.
# - `returnType`: Determines the type of output:
#   - `"fitnessValue"`: Returns a calculated fitness value, combining error and the number of selected features (for wrapper-based methods).
# - `"error"`: Returns the model's prediction error rate (for filter-based methods).

svm_fitnessFunction1 <- function(data,seed,Index,perErr,returnType){
  library(e1071)
  library(caret)
  data =  processing_data(data,seed,Index)
  svmfit = svm(
    device_category ~ ., 
    data = data$train_set, 
    kernel='linear', 
    cost = 1, 
    scale = TRUE)
  p <- predict(svmfit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),188,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

svm_fitnessFunction2 <- function(data,seed,Index,perErr,returnType){
  library(e1071)
  library(caret)
  data =  processing_data(data,seed,Index)
  svmfit = svm(
    device_category ~ ., 
    data = data$train_set, 
    kernel='linear', 
    cost = 20, 
    scale = TRUE)
  p <- predict(svmfit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),30,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

nn_fitnessFunction1 <- function(data,seed,Index,perErr,returnType){
  library(nnet)
  library(skimr)
  data = processing_data(data,seed,Index)
  nnfit <- nnet(device_category~.,
                data = data$train_set,
                decay=0.1,
                size=5,
                trace = FALSE)
  p <- predict(nnfit,
             data$test_set[,1:ncol(data$test_set)-1],
             type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),188,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

nn_fitnessFunction2 <- function(data,seed,Index,perErr,returnType){
  library(nnet)
  library(skimr)
  data = processing_data(data,seed,Index)
  nnfit <- nnet(device_category~.,
                data = data$train_set,
                decay=0.05,
                size=5,
                trace = FALSE)
  p <- predict(nnfit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),30,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

dt_fitnessFunction1 <- function(data,seed,Index,perErr,returnType){
  library(caret)
  library(rpart) 
  data = processing_data(data,seed,Index)
  
  #Build decision_tree model
  dtfit <- rpart(
    device_category~.,
    data = data$train_set,
    method = "class",
    cp=0.001,
    minsplit = 1,
    parms = list(split = "information")
  )
  p <- predict(dtfit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),188,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

dt_fitnessFunction2 <- function(data,seed,Index,perErr,returnType){
  library(caret)
  library(rpart) 
  data = processing_data(data,seed,Index)
  
  #Build decision_tree model
  dtfit <- rpart(
    device_category~.,
    data = data$train_set,
    method = "class",
    cp=0,
    minsplit=1,
    maxdepth = 30,
    minbucket = 1,
    parms = list(split = "information")
  )
  p <- predict(dtfit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),30,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

rf_fitnessFunction1 <- function(data,seed,Index,perErr,returnType){
  library(randomForest)
  library(caret)
  data = processing_data(data,seed,Index)
  
  rffit <- randomForest(
    device_category~.,
    data = data$train_set,
    ntree = 500, 
    mtry = 10,
    importance = T 
  )
  p <- predict(rffit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),188,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

rf_fitnessFunction2 <- function(data,seed,Index,perErr,returnType){
  library(randomForest)
  library(caret)
  data = processing_data(data,seed,Index)
  rffit <- randomForest(
    device_category~.,
    data = data$train_set,
    ntree = 500, 
    mtry = 11,
    importance = T 
  )
  p <- predict(rffit,
               data$test_set[,1:ncol(data$test_set)-1],
               type='class')
  if(returnType == "fitnessValue"){
    return(fitnessFunction(data,as_tibble(p),30,perErr))
  }else{
    accuracy <- mean(p==data$test_set$device_category)
    error <- 1 - accuracy
    return(error)
  }
}

# processing_data: Prepares the dataset by selecting features based on the provided index and splitting it into training and testing sets.
# 
# Parameters:
# - `data`: The input dataset containing features and the target variable (`device_category`).
# - `seed`: The seed value used to ensure reproducibility when splitting the dataset into training and testing sets.
# - `Index`: A binary vector indicating the selected features. `1` indicates the feature is selected, and `0` indicates the feature is not selected.
processing_data <-function(data,seed,Index){
  data <- as_tibble(data)
  #Removes features where `Index[i] == 0` to create a subset of the dataset based on selected features.
  for (i in (ncol(data)-1):1){
    if (Index[i] == 0)
      data <- data[,-i]
  }
  data$device_category = factor(data$device_category)
  # Uses the provided `seed` to split the data into:
  #    - `train_set`: 70% of the data, used for training.
  #    - `test_set`: 30% of the data, used for testing.
  set.seed(seed)
  sample <- sample(x=c(TRUE, FALSE),size= nrow(data), replace=TRUE, prob=c(0.7,0.3))
  train_set <- data[sample, ]
  test_set <- data[!sample, ]
  data <- list(
    train_set = train_set,
    test_set = test_set
  )
  return(data)
}

# fitnessFunction: Calculates the fitness value for a model based on prediction performance and feature selection.
# 
# Parameters:
# - `data`: The processed dataset containing training and testing sets.
# - `p`: The predictions made by the model on the test set.
# - `dime`: The total number of features in the original dataset.
# - `perErr`: The weight assigned to the error rate when calculating the fitness value.
fitnessFunction <- function(data,p,dime,perErr){
  accuracy = mean(p==data$test_set[,ncol(data$test_set)])
  error = 1- accuracy
  #Computes the fitness value using a weighted formula:
  #    - `perErr * error` gives the weighted contribution of the error rate.
  #    - `(1 - perErr) * (number of selected features / total features)` penalizes the use of more features.
  fitnessValue = perErr*error+(1-perErr)*(ncol(data$test_set)-1)/dime
  return (fitnessValue)
}

fitnessValue <- function(dime,numOfSF,accuracy,perErr){
  error = 1- accuracy
  fitnessValue = perErr*error+(1-perErr)*(numOfSF-1)/dime
  return (fitnessValue)
}

# Function :feature_set_summary
# Purpose:to summarize the results of the feature selection algorithm
# This function generates a summary of the best feature subset found during the optimization process.
# The summary includes the optimizer used, the algorithm name, the number of selected features, the
# smallest error achieved, and the selected feature indices.
#
# Parameters:
#   - perErr: The weight assigned to the error rate when calculating the fitness value. It is used to scale the fitness function.
#   - optimizer: The name of the optimizer algorithm used for feature selection (e.g., "BGWO", "BGA","BPSO").
#   - fitAlpha: The fitness value of the best solution found. It is adjusted by the weight of the error rate and the number of features.
#   - dime: The total number of features in the dataset (excluding the target variable).
#   - featAlpha: The binary vector of selected features, where 1 indicates the feature is selected, and 0 means it is not.
#   - algoName: The name of the classifier (e.g., "svm", "nn","dt","rf").
#
# Returns:
#   - bestResult: A list containing:
#       - optimizer: The optimizer used for feature selection.
#       - algorithm: The classifier used.
#       - numAlphaFeature: The number of features selected in the best solution.
#       - smallestError: The smallest error achieved (after adjustment using perErr).
#       - alphaFeatSetIndex: The binary index of the selected features.
feature_set_summary <- function(perErr,optimizer,fitAlpha,dime,featAlpha,algoName){
  fitAlpha = (as.numeric(fitAlpha)-(1-perErr)*sum(featAlpha)/dime)/perErr
  bestResult = list(
    optimizer = optimizer,
    algorithm = algoName,
    numAlphaFeature = sum(featAlpha),
    smallestError = as.numeric(fitAlpha),
    alphaFeatSetIndex = featAlpha
  )
  return(bestResult)
}

### classifier function which return confusion matrix #######
rf_classifier_d <- function(dataD,seed,indexD){
  library(randomForest)
  library(caret)
  dataD = processing_data(dataD,seed,indexD)
  
  rffit <- randomForest(
    device_category~.,
    data = dataD$train_set,
    ntree = 500, 
    mtry = 10,
    importance = T 
  )
  p =predict(rffit,dataD$test_set[,1:(ncol(dataD$test_set)-1)], type='class')
  t = table(p,dataD$test_set$device_category)
  return (t)
}

### classifier function which return confusion matrix #######
rf_classifier_s <- function(dataS,seed,indexS){
  library(randomForest)
  library(caret)
  dataS = processing_data(dataS,seed,indexS)
  
  rffit <- randomForest(
    device_category~.,
    data = dataS$train_set,
    ntree = 500, 
    mtry = 11,
    importance = T 
  )
  p =predict(rffit,dataS$test_set[,1:(ncol(dataS$test_set)-1)], type='class')
  t = table(p,dataS$test_set$device_category)
  return (t)
}