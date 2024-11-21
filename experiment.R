# Experiment Execution Script for "Machine Learning for IoT Device Identification: A Comparative Study"
#
# This script conducts a series of experiments on two IoT datasets ('IotDataset.csv' and 'IotSentinal.csv') 
# using various feature selection methods and classification algorithms. The primary goal is to evaluate the 
# performance of different classifiers (SVM, Neural Networks, Decision Trees, Random Forest) on feature sets 
# derived from the IoT datasets using different feature selection techniques:
# 1. Pearson Correlation Coefficient (PCC)
# 2. Mutual Information (MI)
# 3. Binary Grey Wolf Optimizer (BGWO)
# 4. Binary Genetic Algorithm (BGA)
# 5. Binary Particle Swarm Optimization (BPSO)
#
# The experiment follows these key steps:
# - Loading and initializing the datasets.
# - Performing feature selection using the aforementioned techniques (PCC, MI, BGWO, BGA, BPSO).
# - Training and evaluating classifiers on the selected feature sets.
# - Repeating the experiments with multiple random seeds to ensure statistical significance.
# - Storing results for comparison and further analysis.
#
# The script is organized into the following sections:
# 1. Full Feature Set Evaluation: Evaluates classifiers using all available features.
# 2. Pearson Correlation Coefficient (PCC): Features are selected based on correlation thresholds.
# 3. Mutual Information (MI): Features are selected based on mutual information thresholds.
# 4. Binary Grey Wolf Optimizer (BGWO): Optimized feature selection and classification.
# 5. Binary Genetic Algorithm (BGA): Optimized feature selection and classification.
# 6. Binary Particle Swarm Optimization (BPSO): Optimized feature selection and classification.
#
# Results are saved at each stage, enabling easy comparison between different feature selection methods and classifiers.

library(purrr)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(GA)
library(e1071)
library(caret)
library(beepr)
library(dplyr)
library(ggrepel)
library(infotheo)
library(mlr)
library(FSelector)

source("featureSelectionMethods.R") # Custom feature selection methods
source("classifier.R") # Classifier-related functions
source('drawGraph.R') # Graph plotting functions
source('otherFunctions.R') # Additional utility functions

## Parameter initialization and dataset preparation
# Load Dataset 1
dataD <- read.csv('IotDataset.csv')
dimeD = ncol(dataD)-1 # Number of features (exclude label)
indexD <- data.frame(  # Placeholder for indices
  matrix(data = 1,
         ncol = 188)
)

# Load Dataset 2
dataS <- read.csv('IotSentinal.csv')
dimeS = ncol(dataS)-1
indexS <- data.frame(
  matrix(data = 1,
         ncol = ncol(dataS)-1)
)

# Set experiment parameters
bootnum = 10 # Number of bootstrap iterations
seed = seq(1,bootnum,1) # Random seeds for reproducibility
perErr=0.99 # The weight assigned to the error rate when calculating the fitness value

#########	Full Feature Set #############
#########	Full Feature Set #############
#########	Full Feature Set #############
## Evaluate errors using all features in both datasets
# Dataset 1: Full feature error calculations using SVM, NN, DT, and RF
error_allFeatures_svm_D <- map(seed,function(seed){
  svm_fitnessFunction1(dataD,seed,indexD,perErr,"error")
})
error_allFeatures_nn_D <- map(seed,function(seed){
  nn_fitnessFunction1(dataD,seed,indexD,perErr,"error")
})
error_allFeatures_dt_D <- map(seed,function(seed){
  
  dt_fitnessFunction1(dataD,seed,indexD,perErr,"error")
})
error_allFeatures_rf_D <- map(seed,function(seed){
  rf_fitnessFunction1(dataD,seed,indexD,perErr,"error")
})

# Dataset 2: Full feature error calculations
error_allFeatures_svm_S <- map(seed,function(seed){
  svm_fitnessFunction2(dataS,seed,indexS,perErr,"error")
})
error_allFeatures_nn_S <- map(seed,function(seed){
  nn_fitnessFunction2(dataS,seed,indexS,perErr,"error")
})
error_allFeatures_dt_S <- map(seed,function(seed){
  dt_fitnessFunction2(dataS,seed,indexS,perErr,"error")
})
error_allFeatures_rf_S <- map(seed,function(seed){
  rf_fitnessFunction2(dataS,seed,indexS,perErr,"error")
})

# Organize and save results for full-feature evaluations
original_iotD <-formatConvert_org("Dataset1",dimeD,error_allFeatures_svm_D,error_allFeatures_nn_D,error_allFeatures_dt_D,error_allFeatures_rf_D)
original_iotS <-formatConvert_org("Dataset2",dimeS,error_allFeatures_svm_S,error_allFeatures_nn_S,error_allFeatures_dt_S,error_allFeatures_rf_S)

save(original_iotD,original_iotS, file = "originalResultDS.RData")

#########	Pearson Correlation Coefficient (PCC)#############
#########	Pearson Correlation Coefficient (PCC)#############
#########	Pearson Correlation Coefficient (PCC)#############
## Feature selection using Pearson Correlation Coefficient (PCC)
# Define Pearson correlation thresholds
cutoff = c(0.9,0.8,0.7,0.6,0.5)

# Dataset 1: Perform PCC-based feature selection and accuracy evaluation
PCC_accD_svm = PCC_accD_nn = PCC_accD_dt = PCC_accD_rf <- as.data.frame(
  matrix(
    data = NA,
    ncol = 5,
    nrow = 0,
    dimnames = list(c(),
                    c("dataset","classifierName","cutoff","numOfSelectedFea","accuracy"))
  )
)

for( i in seed){   # Loop through seeds
  for( j in cutoff){ # Loop through Pearson correlation thresholds
    PCC_result_svm  = as.data.frame(pearsonCorrelation(dataD,i,svm_fitnessFunction1,"SVM",j,"Dataset1"))
    PCC_accD_svm = rbind(PCC_accD_svm,PCC_result_svm)
    PCC_result_nn = as.data.frame(pearsonCorrelation(dataD,i,nn_fitnessFunction1,"Neural Networks",j,"Dataset1"))
    PCC_accD_nn = rbind(PCC_accD_nn,PCC_result_nn)
    PCC_result_dt = as.data.frame(pearsonCorrelation(dataD,i,dt_fitnessFunction1,"Decision Tree",j,"Dataset1"))
    PCC_accD_dt = rbind(PCC_accD_dt,PCC_result_dt)
    PCC_result_rf = as.data.frame(pearsonCorrelation(dataD,i,rf_fitnessFunction1,"Random Forest",j,"Dataset1"))
    PCC_accD_rf = rbind(PCC_accD_rf,PCC_result_rf)
  }
}

# Summarize PCC results for Dataset 1
PCC_aveD_svm = count_average_new(PCC_accD_svm)
PCC_aveD_nn = count_average_new(PCC_accD_nn)
PCC_aveD_dt = count_average_new(PCC_accD_dt)
PCC_aveD_rf = count_average_new(PCC_accD_rf)
PCC_aveD = rbind(PCC_aveD_svm,PCC_aveD_nn,PCC_aveD_dt,PCC_aveD_rf)

# Dataset 2: Repeat PCC-based feature selection for the second dataset (similar to above)...
PCC_accS_svm = PCC_accS_nn = PCC_accS_dt = PCC_accS_rf <- as.data.frame(
  matrix(
    data = NA,
    ncol = 5,
    nrow = 0,
    dimnames = list(c(),
                    c("dataset","classifierName","cutoff","numOfSelectedFea","accuracy"))
  )
  
)

for( i in seed){
  for( j in cutoff){
    PCC_result_svm  = as.data.frame(pearsonCorrelation(dataS,i,svm_fitnessFunction2,"SVM",j,"Dataset2"))
    PCC_accS_svm = rbind(PCC_accS_svm,PCC_result_svm)
    PCC_result_nn = as.data.frame(pearsonCorrelation(dataS,i,nn_fitnessFunction2,"Neural Networks",j,"Dataset2"))
    PCC_accS_nn = rbind(PCC_accS_nn,PCC_result_nn)
    PCC_result_dt = as.data.frame(pearsonCorrelation(dataS,i,dt_fitnessFunction2,"Decision Tree",j,"Dataset2"))
    PCC_accS_dt = rbind(PCC_accS_dt,PCC_result_dt)
    PCC_result_rf = as.data.frame(pearsonCorrelation(dataS,i,rf_fitnessFunction2,"Random Forest",j,"Dataset22"))
    PCC_accS_rf = rbind(PCC_accS_rf,PCC_result_rf)
  }
}

# Summarize PCC results for Dataset 2
PCC_aveS_svm = count_average_new(PCC_accS_svm)
PCC_aveS_nn = count_average_new(PCC_accS_nn)
PCC_aveS_dt = count_average_new(PCC_accS_dt)
PCC_aveS_rf = count_average_new(PCC_accS_rf)
PCC_aveS = rbind(PCC_aveS_svm,PCC_aveS_nn,PCC_aveS_dt,PCC_aveS_rf)

# Visualize results
# Draw Figure 6. Results of Dataset1 gained by using PCC-based feature selection
drawPearsonResult(PCC_aveD,188)
# Draw Figure 7. Results of Dataset 2 gained by using PCC-based feature selection
drawPearsonResult(PCC_aveS,30)

# Find the best PCC results for each datasets 
# Dataset 1
bestPCCResultD <- findBestResult(PCC_aveD,dimeD,perErr)
# Dataset 2
bestPCCResultS <- findBestResult(PCC_aveS,dimeS,perErr)

## Save the best PCC results
save(bestPCCResultD,file = "bestPCCResultD.RData")
save(bestPCCResultS,file = "bestPCCResultS.RData")

# Organize the result of all experiments
# Combine experiment results for all classifiers and 
# find the best result for each classifier in each experiment based on the highest fitness value.
PCC_accD <- rbind(PCC_accD_svm,PCC_accD_nn,PCC_accD_dt,PCC_accD_rf)
PCC_accS <- rbind(PCC_accS_svm,PCC_accS_nn,PCC_accS_dt,PCC_accS_rf)
PCC_iotD <- findBestResult2("PCC",PCC_accD,dimeD,perErr,length(seed),length(cutoff))
PCC_iotS <- findBestResult2("PCC",PCC_accS,dimeS,perErr,length(seed),length(cutoff))
save(PCC_iotD,PCC_iotS,file = "bestPCCResultNew.RData")


#########	Mutual Information (MI)#############
#########	Mutual Information (MI)#############
#########	Mutual Information (MI)#############
## Feature selection using Mutual Information (MI)
# Define boundaries for mutual information thresholds
boundary = c(0.5,0.4,0.3,0.2,0.1,0.01)

########### MI-Dataset1 ################
#Initialize empty data frames to store results for each classifier
MI_accD_svm = MI_accD_nn = MI_accD_dt = MI_accD_rf <- as.data.frame(
  matrix(
    data = NA,
    ncol = 5,
    nrow = 0,
    dimnames = list(c(),
                    c("dataset","classifierName","boundary","numOfSelectedFea","accuracy"))
  )
)

for( i in seed){  # Loop through seed values
  for( j in boundary){ # Loop through different boundary values (thresholds for MI)
    MI_result_svm  = as.data.frame(mutualInfo(dataD,i,svm_fitnessFunction1,"SVM",j,"Dataset1"))
    MI_accD_svm = rbind(MI_accD_svm,MI_result_svm)
    MI_result_nn = as.data.frame(mutualInfo(dataD,i,nn_fitnessFunction1,"Neural Networks",j,"Dataset1"))
    MI_accD_nn = rbind(MI_accD_nn,MI_result_nn)
    MI_result_dt = as.data.frame(mutualInfo(dataD,i,dt_fitnessFunction1,"Decision Tree",j,"Dataset1"))
    MI_accD_dt = rbind(MI_accD_dt,MI_result_dt)
    MI_result_rf = as.data.frame(mutualInfo(dataD,i,rf_fitnessFunction1,"Random Forest",j,"Dataset1"))
    MI_accD_rf = rbind(MI_accD_rf,MI_result_rf)
  }
}

## Calculate averages for each classifier and summarize the results
MI_aveD_svm = count_average_new(MI_accD_svm)
MI_aveD_nn = count_average_new(MI_accD_nn)
MI_aveD_dt = count_average_new(MI_accD_dt)
MI_aveD_rf = count_average_new(MI_accD_rf)
MI_aveD = rbind(MI_aveD_svm,MI_aveD_nn,MI_aveD_dt,MI_aveD_rf)

# Repeat MI-based feature selection for the second dataset (similar to above)...
########### MI-Dataset2 ################
MI_accS_svm = MI_accS_nn = MI_accS_dt = MI_accS_rf <- as.data.frame(
  matrix(
    data = NA,
    ncol = 5,
    nrow = 0,
    dimnames = list(c(),
                    c("dataset","classifierName","boundary","numOfSelectedFea","accuracy"))
  )
  
)

for( i in seed){
  for( j in boundary){
    MI_result_svm  = as.data.frame(mutualInfo(dataS,i,svm_fitnessFunction2,"SVM",j,"Dataset2"))
    MI_accS_svm = rbind(MI_accS_svm,MI_result_svm)
    MI_result_nn = as.data.frame(mutualInfo(dataS,i,nn_fitnessFunction2,"Neural Networks",j,"Dataset2"))
    MI_accS_nn = rbind(MI_accS_nn,MI_result_nn)
    MI_result_dt = as.data.frame(mutualInfo(dataS,i,dt_fitnessFunction2,"Decision Tree",j,"Dataset2"))
    MI_accS_dt = rbind(MI_accS_dt,MI_result_dt)
    MI_result_rf = as.data.frame(mutualInfo(dataS,i,rf_fitnessFunction2,"Random Forest",j,"Dataset22"))
    MI_accS_rf = rbind(MI_accS_rf,MI_result_rf)
  }
}

## result summarization
MI_aveS_svm = count_average_new(MI_accS_svm)
MI_aveS_nn = count_average_new(MI_accS_nn)
MI_aveS_dt = count_average_new(MI_accS_dt)
MI_aveS_rf = count_average_new(MI_accS_rf)
MI_aveS = rbind(MI_aveS_svm,MI_aveS_nn,MI_aveS_dt,MI_aveS_rf)

# result visualization
# Draw Figure 8 Results of Dataset 1 gained by using MI-based feature selection
drawMutualInfoResult(MI_aveD,188)
# Draw Figure 9 Results of Dataset 2 gained by using MI-based feature selection
drawMutualInfoResult(MI_aveS,30)

# Find the boundary value that provides the maximum fitness
# dataset 1
bestMIResultD <- findBestResult(MI_aveD,dimeD,perErr)
# dataset 2
bestMIResultS <- findBestResult(MI_aveS,dimeS,perErr)

# Save the best average result from MI-based feature selection
save(bestMIResultD,file = "bestMIResultD.RData")
save(bestMIResultS,file = "bestMIResultS.RData")

# Organize the results of all experiments
MI_accD <- rbind(MI_accD_svm,MI_accD_nn,MI_accD_dt,MI_accD_rf)
MI_accS <- rbind(MI_accS_svm,MI_accS_nn,MI_accS_dt,MI_accS_rf)
MI_iotD <- findBestResult2("MI",MI_accD,dimeD,perErr,length(seed),length(boundary))
MI_iotS <- findBestResult2("MI",MI_accS,dimeS,perErr,length(seed),length(boundary))
save(MI_iotD,MI_iotS,file = "bestMIResultNew.RData")

####################BGWO#################
####################BGWO#################
####################BGWO#################
############# BGWO-Dataset1 #############
# Define parameters for the BGWO algorithm
N_wolf = 10 # Number of wolves
max_Iter = 100 # Maximum number of iterations

# Apply BGWO algorithm for each classifier and dataset using different seeds
svm_BGWO_iotD <- map(seed,function(seed){
  # Perform BGWO for SVM classifier on Dataset 1
  BGWO(dataD,seed,N_wolf,max_Iter,svm_fitnessFunction1,"svm",perErr)
})

nn_BGWO_iotD <- map(seed,function(seed){
  # Perform BGWO for Neural Networks classifier on Dataset 1
  BGWO(dataD,seed,N_wolf,max_Iter,nn_fitnessFunction1,"nn",perErr)
})

dt_BGWO_iotD <- map(seed,function(seed){
  # Perform BGWO for Decision Tree classifier on Dataset 1
  BGWO(dataD,seed,N_wolf,max_Iter,dt_fitnessFunction1,"dt",perErr)
})

rf_BGWO_iotD <- map(seed,function(seed){
  # Perform BGWO for Random Forest classifier on Dataset 1
  BGWO(dataD,seed,N_wolf,max_Iter,rf_fitnessFunction1,"rf",perErr)
})

## Summarize the results for Dataset 1
BGWO_iotD = formatConvert("Dataset1",svm_BGWO_iotD,nn_BGWO_iotD,dt_BGWO_iotD,rf_BGWO_iotD)

# Calculate the average performance (accuracy) for each classifier on Dataset 1
svm_BGWO_iotD_average = count_average(svm_BGWO_iotD,error_allFeatures_svm_D)
nn_BGWO_iotD_average = count_average(nn_BGWO_iotD,error_allFeatures_nn_D)
dt_BGWO_iotD_average = count_average(dt_BGWO_iotD,error_allFeatures_dt_D)
rf_BGWO_iotD_average = count_average(rf_BGWO_iotD,error_allFeatures_rf_D)

# Format the average results for Dataset 1
bestBGWOResultD = aveFSFormatConvert("Dataset1",
                                     svm_BGWO_iotD_average,
                                     nn_BGWO_iotD_average,
                                     dt_BGWO_iotD_average,
                                     rf_BGWO_iotD_average)
# Save the results for Dataset 1
save(BGWO_iotD,bestBGWOResultD,file = "bestBGWOResultD.RData")


################# BGWO-Dataset2 #############
# Apply BGWO algorithm for each classifier and dataset using different seeds for Dataset 2
svm_BGWO_iotS <- map(seed,function(seed){
  BGWO(dataS,seed,N_wolf,max_Iter,svm_fitnessFunction2,"svm",perErr)
})

nn_BGWO_iotS <- map(seed,function(seed){
  BGWO(dataS,seed,N_wolf,max_Iter,nn_fitnessFunction2,"nn",perErr)
})

dt_BGWO_iotS <- map(seed,function(seed){
  BGWO(dataS,seed,N_wolf,max_Iter,dt_fitnessFunction2,"dt",perErr)
})

rf_BGWO_iotS <- map(seed,function(seed){
  BGWO(dataS,seed,N_wolf,max_Iter,rf_fitnessFunction2,"rf",perErr)
})

## Summarize the results for Dataset 2
BGWO_iotS = formatConvert("Dataset2",svm_BGWO_iotS,nn_BGWO_iotS,dt_BGWO_iotS,rf_BGWO_iotS)
svm_BGWO_iotS_average = count_average(svm_BGWO_iotS,error_allFeatures_svm_S)
nn_BGWO_iotS_average = count_average(nn_BGWO_iotS,error_allFeatures_nn_S)
dt_BGWO_iotS_average = count_average(dt_BGWO_iotS,error_allFeatures_dt_S)
rf_BGWO_iotS_average = count_average(rf_BGWO_iotS,error_allFeatures_rf_S)

# Format the average results for Dataset 2
bestBGWOResultS = aveFSFormatConvert("Dataset2",
                                       svm_BGWO_iotS_average,
                                       nn_BGWO_iotS_average,
                                       dt_BGWO_iotS_average,
                                       rf_BGWO_iotS_average)
# Save the results for Dataset 2
save(BGWO_iotS,bestBGWOResultS,file = "bestBGWOResultS.RData")


################ GA ##################
################ GA ##################
################ GA ##################
########## GA-Dataset 1 #############
# Define parameters for the Genetic Algorithm (GA)
population_size = 10 # Population size (number of individuals in each generation)
max_Iter = 100 # Maximum number of iterations (generations)
crossover = 0.8 # Crossover probability (the probability that two parents will combine to create offspring)
mutation = 0.1 # Mutation probability (the probability that an individual will mutate)

# Apply GA algorithm for each classifier and dataset using different seeds
svm_BGA_iotD <- map(seed,function(seed){
  BGA(dataD,seed,population_size,max_Iter,svm_fitnessFunction1,"svm",perErr,crossover,mutation)
})

nn_BGA_iotD <- map(seed,function(seed){
  BGA(dataD,seed,population_size,max_Iter,nn_fitnessFunction1,"nn",perErr,crossover,mutation)
})

dt_BGA_iotD <- map(seed,function(seed){
  BGA(dataD,seed,population_size,max_Iter,dt_fitnessFunction1,"dt",perErr,crossover,mutation)
})

rf_BGA_iotD <- map(seed,function(seed){
  BGA(dataD,seed,population_size,max_Iter,rf_fitnessFunction1,"rf",perErr,crossover,mutation)
})

## Summarize the results for Dataset 1
BGA_iotD = formatConvert("Dataset1",svm_BGA_iotD,nn_BGA_iotD,dt_BGA_iotD,rf_BGA_iotD)


# Calculate the average performance (accuracy) for each classifier on Dataset 1
svm_BGA_iotD_average = count_average(svm_BGA_iotD,error_allFeatures_svm_D)
nn_BGA_iotD_average = count_average(nn_BGA_iotD,error_allFeatures_nn_D)
dt_BGA_iotD_average = count_average(dt_BGA_iotD,error_allFeatures_dt_D)
rf_BGA_iotD_average = count_average(rf_BGA_iotD,error_allFeatures_rf_D)

# Format the average results for Dataset 1
bestBGAResultD = aveFSFormatConvert("Dataset1",
                                    svm_BGA_iotD_average,
                                    nn_BGA_iotD_average,
                                    dt_BGA_iotD_average,
                                    rf_BGA_iotD_average)

# Save the results for Dataset 1
save(BGA_iotD,bestBGAResultD, file = "bestBGAResultD.RData")


########## GA-Dataset2 #############
# Apply GA algorithm for each classifier and dataset 
svm_BGA_iotS <- map(seed,function(seed){
  BGA(dataS,seed,population_size,max_Iter,svm_fitnessFunction2,"svm",perErr,crossover,mutation)
})

nn_BGA_iotS <- map(seed,function(seed){
  BGA(dataS,seed,population_size,max_Iter,nn_fitnessFunction2,"nn",perErr,crossover,mutation)
})

dt_BGA_iotS <- map(seed,function(seed){
  BGA(dataS,seed,population_size,max_Iter,dt_fitnessFunction2,"dt",perErr,crossover,mutation)
})

rf_BGA_iotS <- map(seed,function(seed){
  BGA(dataS,seed,population_size,max_Iter,rf_fitnessFunction2,"rf",perErr,crossover,mutation)
})

# Summarize the results for Dataset 2
BGA_iotS = formatConvert("Dataset2",svm_BGA_iotS,nn_BGA_iotS,dt_BGA_iotS,rf_BGA_iotS)

# Calculate the average performance (accuracy) for each classifier on Dataset 2
svm_BGA_iotS_average = count_average(svm_BGA_iotS,error_allFeatures_svm_S)
nn_BGA_iotS_average = count_average(nn_BGA_iotS,error_allFeatures_nn_S)
dt_BGA_iotS_average = count_average(dt_BGA_iotS,error_allFeatures_dt_S)
rf_BGA_iotS_average = count_average(rf_BGA_iotS,error_allFeatures_rf_S)

# Format the average results for Dataset 2
bestBGAResultS = aveFSFormatConvert("Dataset2",
                                    svm_BGA_iotS_average,
                                    nn_BGA_iotS_average,
                                    dt_BGA_iotS_average,
                                    rf_BGA_iotS_average)
# Save the results for Dataset 2
save(BGA_iotS,bestBGAResultS, file = "bestBGAResultS.RData")

##############BPSO##################
##############BPSO##################
##############BPSO##################
########### BPSO-Dataset1 ##############
# Define parameters for the Binary Particle Swarm Optimization (BPSO) on Dataset 1
n_particles = 10  # Number of particles (individuals in the swarm)
max_Iter = 100  # Maximum number of iterations (steps of optimization)
inertia = 0.5  # Inertia weight (controls the impact of the previous velocity on the current velocity)
c1 = 0.1  # Cognitive coefficient (affects the particle's attraction to its personal best position)
c2 = 0.1  # Social coefficient (affects the particle's attraction to the swarm's global best position)
Vmax = 5  # Maximum velocity (controls how large the change can be in one step)

# Apply BPSO for each classifier on Dataset 1 
svm_BPSO_iotD <- map(seed,function(seed){
  BPSO(dataD, seed, n_particles, max_Iter, svm_fitnessFunction1, "svm", perErr, inertia, c1, c2, Vmax)
})

nn_BPSO_iotD <- map(seed,function(seed){
  BPSO(dataD, seed, n_particles, max_Iter, nn_fitnessFunction1, "nn", perErr, inertia, c1, c2, Vmax)
})

dt_BPSO_iotD <- map(seed,function(seed){
  BPSO(dataD, seed, n_particles, max_Iter, dt_fitnessFunction1, "dt", perErr, inertia, c1, c2, Vmax)
})

rf_BPSO_iotD <- map(seed,function(seed){
  BPSO(dataD, seed, n_particles, max_Iter, rf_fitnessFunction1, "rf", perErr, inertia, c1, c2, Vmax)
})

# Summarize the results for Dataset 1
BPSO_iotD = formatConvert("Dataset1",svm_BPSO_iotD,nn_BPSO_iotD,dt_BPSO_iotD,rf_BPSO_iotD)
# Calculate the average performance for each classifier on Dataset 1
svm_BPSO_iotD_average = count_average(svm_BPSO_iotD,error_allFeatures_svm_D)
nn_BPSO_iotD_average = count_average(nn_BPSO_iotD,error_allFeatures_nn_D)
dt_BPSO_iotD_average = count_average(dt_BPSO_iotD,error_allFeatures_dt_D)
rf_BPSO_iotD_average = count_average(rf_BPSO_iotD,error_allFeatures_rf_D)

# Format the average results for Dataset 1
bestBPSOResultD = aveFSFormatConvert("Dataset1",
                                     svm_BPSO_iotD_average,
                                     nn_BPSO_iotD_average,
                                     dt_BPSO_iotD_average,
                                     rf_BPSO_iotD_average)
save(BPSO_iotD,bestBPSOResultD, file = "bestBPSOResultD.RData")

########### BPSO-Dataset2 ##############
# Apply BPSO for each classifier on Dataset 2 using different random seeds
svm_BPSO_iotS <- map(seed,function(seed){
  BPSO(dataS, seed, n_particles, max_Iter, svm_fitnessFunction2, "svm", perErr, inertia, c1, c2, Vmax)
})

nn_BPSO_iotS <- map(seed,function(seed){
  BPSO(dataS, seed, n_particles, max_Iter, nn_fitnessFunction2, "nn", perErr, inertia, c1, c2, Vmax)
})

dt_BPSO_iotS <- map(seed,function(seed){
  BPSO(dataS, seed, n_particles, max_Iter, dt_fitnessFunction2, "dt", perErr, inertia, c1, c2, Vmax)
})

rf_BPSO_iotS <- map(seed,function(seed){
  BPSO(dataS, seed, n_particles, max_Iter, rf_fitnessFunction2, "rf", perErr, inertia, c1, c2, Vmax)
})

# Summarize the results for Dataset 2
BPSO_iotS = formatConvert("Dataset2",svm_BPSO_iotS,nn_BPSO_iotS,dt_BPSO_iotS,rf_BPSO_iotS)

# Calculate the average performance (error rate or accuracy) for each classifier on Dataset 2
svm_BPSO_iotS_average = count_average(svm_BPSO_iotS,error_allFeatures_svm_S)
nn_BPSO_iotS_average = count_average(nn_BPSO_iotS,error_allFeatures_nn_S)
dt_BPSO_iotS_average = count_average(dt_BPSO_iotS,error_allFeatures_dt_S)
rf_BPSO_iotS_average = count_average(rf_BPSO_iotS,error_allFeatures_rf_S)

# Format the average results for Dataset 2
bestBPSOResultS = aveFSFormatConvert("Dataset2",
                                     svm_BPSO_iotS_average,
                                     nn_BPSO_iotS_average,
                                     dt_BPSO_iotS_average,
                                     rf_BPSO_iotS_average)
save(BPSO_iotS,bestBPSOResultS, file = "bestBPSOResultS.RData")



######### result visualization ######
######### Figure 10 - Figure 15 #####
############# Dataset 1 #############
drawResult(svm_BGWO_iotD,error_allFeatures_svm_D,"SVM","Dataset1",dimeD)
drawResult(nn_BGWO_iotD,error_allFeatures_nn_D,"Neural Networks","Dataset1",dimeD)
drawResult(dt_BGWO_iotD,error_allFeatures_dt_D,"Decision Tree","Dataset1",dimeD)
drawResult(rf_BGWO_iotD,error_allFeatures_rf_D,"Random Forest","Dataset1",dimeD)
############# Dataset 2 #############
drawResult(svm_BGWO_iotS,error_allFeatures_svm_S,"SVM","Dataset2",30)
drawResult(nn_BGWO_iotS,error_allFeatures_nn_S,"Neural Networks","Dataset2",30)
drawResult(dt_BGWO_iotS,error_allFeatures_dt_S,"Decision Tree","Dataset2",30)
drawResult(rf_BGWO_iotS,error_allFeatures_rf_S,"Random Forest","Dataset2",30)

#############Draw Figure 16 Comparison between FS methods and classifiers in Dataset 1 #######
experiment_result_D <- rbind(original_iotD,PCC_iotD,MI_iotD,BGA_iotD,BPSO_iotD,BGWO_iotD)
drawResultSumNew(experiment_result_D)

#############Draw Figure 17 Comparison between FS methods and classifiers in Dataset 2 #######
experiment_result_S <- rbind(original_iotS,PCC_iotS,MI_iotS,BGA_iotS,BPSO_iotS,BGWO_iotS)
drawResultSumNew(experiment_result_S)

######### Draw confusion matrix of the best result #########
###################### Dataset 1 ###########################
featIndex_su_D_rf_9 <- rf_BGWO_iotD[[9]]$alphaFeatSetIndex
p_rf_D <- rf_classifier_d(dataD,seed = 9,indexD = featIndex_su_D_rf_9)
######### Draw Figure 18 ##########
drawConfusionMatrix(p_rf_D,"Dataset1")

###################### Dataset 2 ###########################
featIndex_su_S_rf_8 <- rf_BGWO_iotS[[8]]$alphaFeatSetIndex
p_rf_S <- rf_classifier_s(dataS,seed = 8,indexS = featIndex_su_S_rf_8)
######### Draw Figure 19 ##########
drawConfusionMatrix(p_rf_S,"Dataset2")
