#count_average Function
#Purpose: Calculates the average optimization results including average error, number of features...
count_average <- function(optresult,completeSetAcc){
  len = length(optresult)
  smallestErr = 0
  numFeat = 0
  compeleteFeatSetAcc = 0
  for (i in 1:len){
    smallestErr = smallestErr+optresult[[i]]$smallestError
    numFeat = numFeat+optresult[[i]]$numAlphaFeature
    compeleteFeatSetAcc = compeleteFeatSetAcc + completeSetAcc[[i]]
  }
  totalFeature = length(optresult[[1]]$alphaFeatSetIndex)
  featNum = data.frame(
    matrix(
      data=0,
      ncol = totalFeature,
      dimnames = list(c(),
                      c(colnames(optresult[[1]]$alphaFeatSetIndex)))
    )
  )
  
  for(i in 1:totalFeature){
    for(j in 1:len){
      if(optresult[[j]]$alphaFeatSetIndex[i] == 1){
        featNum[i] = featNum[i] +1
      }
    }
  }
  numFeatSelectedByAll = 0
  for(i in 1:totalFeature){
    if(featNum[i] == len){
      featNum[i] = 1
      numFeatSelectedByAll = numFeatSelectedByAll+1
    }else{
      featNum[i] = 0
    }
  }
  average = list(
    optimizer = optresult[[1]]$optimizer,
    algorithm = optresult[[1]]$algorithm,
    numFeat = numFeat/len,
    compeleteFeatSetAcc = 1-compeleteFeatSetAcc/len,
    FSSelectedHighestAcc = 1-smallestErr/len,
    numFeatSelectedByAll = numFeatSelectedByAll,
    optFeatIndex = featNum
  )
  return(average)
}

#count_average_new Function
#Purpose: Calculates the average number of features and accuracy for filter-based FS methods.
count_average_new <- function(accdf){
  averageResult <- aggregate(accdf$numOfSelectedFea,by = list(accdf$dataset,accdf$classifierName,accdf[[3]]),FUN = mean )
  average_accuracy <- aggregate(accdf$accuracy,by = list(accdf$dataset,accdf$classifierName,accdf[[3]]),FUN = mean )
  averageResult$average_accuracy = average_accuracy$x
  colnames(averageResult) = c(colnames(accdf[1:3]),"argNumOfSelectedFea","argAccuracy")
  return(averageResult)
}

# Function to find the best fitness value for each classifier
findBestResult <- function(data, totalFeature,perErr) {
  # Calculate the fitness value for all rows
  data$fitnessValue <- data$argAccuracy * perErr + 
    (totalFeature - data$argNumOfSelectedFea) / totalFeature * (1-perErr)
  
  # Group by classifierName and find the row with the maximum fitness value for each classifier
  bestResult <- data %>%
    group_by(classifierName) %>%
    filter(fitnessValue == max(fitnessValue)) %>%
    ungroup() 
  
  return(bestResult)
}

#aveFSFormatConvert Function
#Purpose: Converts average feature selection results into a standardized data frame format.
aveFSFormatConvert <- function(dataset,...){
  ave <- as.data.frame(
    matrix(
      data = NA,
      ncol = 5,
      nrow = 4,
      dimnames = list(c(),
                      c("dataset","classifierName","FSMethod","numOfSelectedFea","accuracy"))
    )
  )
  
  j = 1
  for(i in list(...)){
    ave[j,1] <- dataset
    ave[j,2] <- i$algorithm
    ave[j,3] <- i$optimizer
    ave[j,4] <- i$numFeat
    ave[j,5] <- i$FSSelectedHighestAcc
    j = j+1
  }
  return(ave)
}

#orgFSFormatConvert Function
#Purpose: Formats results of the original feature set into a standardized data frame.
orgFSFormatConvert <- function(dataset,...){
  original <- as.data.frame(
    matrix(
      data = NA,
      ncol = 5,
      nrow = 4,
      dimnames = list(c(),
                      c("dataset","classifierName","FSMethod","numOfSelectedFea","accuracy"))
    )
  )
  
  j = 1
  for(i in list(...)){
    original[j,1] <- dataset
    original[j,2] <- i$algorithm
    original[j,3] <- "Original"
    original[j,4] <- length(i$optFeatIndex)
    original[j,5] <- i$compeleteFeatSetAcc
    j = j+1
  }
  return(original)
}

#format_Convert Function
#Purpose: Converts individual or multiple feature selection results into a standardized data frame format.
format_Convert <- function(dataset,df){
  result <- as.data.frame(
    matrix(
      data = NA,
      ncol = 6,
      nrow = length(df),
      dimnames = list(c(),
                      c("experiment","dataset","classifierName","FSMethod","numOfSelectedFea","accuracy"))
    )
  )
  
  j = 1
  for(i in df){
      result[j,1] <- j
      result[j,2] <- dataset
      result[j,3] <- i$algorithm
      result[j,4] <- i$optimizer
      result[j,5] <-  i$numAlphaFeature
      result[j,6] <- 1 - i$smallestError
      j = j+1
    }
  return(result)
}

#formatConvert Function
#Purpose: Processes and formats multiple feature selection results into a unified format.
formatConvert <- function(dataset,...){
  result <- data.frame()
  for (i in list(...)) {
    result <- rbind(result,format_Convert(dataset,i))
  }
  return(result)
}

#format_Convert_org Function
#Purpose: Formats results of the original feature set, including error and feature count.
format_Convert_org <- function(dataset,algorithm,dime,error_allFeatures){
  result <- as.data.frame(
    matrix(
      data = NA,
      ncol = 6,
      nrow = length(error_allFeatures),
      dimnames = list(c(),
                      c("experiment","dataset","classifierName","FSMethod","numOfSelectedFea","accuracy"))
    ))
  
  j = 1
  for(i in 1:length(error_allFeatures_dt_D)){
    result[j,1] <- j
    result[j,2] <- dataset
    result[j,3] <- algorithm
    result[j,4] <- "ORIGINAL"
    result[j,5] <-  dime
    result[j,6] <- 1 - error_allFeatures[[i]]
    j = j+1
  }
  return(result)
}

#formatConvert_org Function
#Purpose: Combines multiple original feature set results into a standardized format.
formatConvert_org <- function(dataset,dime,error_allFeatures_svm,error_allFeatures_nn,error_allFeatures_dt,error_allFeatures_rf){
  result <- data.frame()
  result <- rbind(result,
                  format_Convert_org(dataset,"svm",dime,error_allFeatures_svm),
                  format_Convert_org(dataset,"nn",dime,error_allFeatures_nn),
                  format_Convert_org(dataset,"dt",dime,error_allFeatures_dt),
                  format_Convert_org(dataset,"rf",dime,error_allFeatures_rf))
  return(result)
}

#findBestResult2 Function
#Purpose: Finds the best result for each classifier in each experiment based on the highest fitness value.
findBestResult2 <- function(FSMethod,acc,dime,perErr,numOfExperiment,numOfBoundary){
  acc$fitnessValue = 1- fitnessValue(dime,acc$numOfSelectedFea,acc$accuracy,perErr)
  acc$experiment <- rep(1:numOfExperiment,each = numOfBoundary,times=4)
  
  # Filter rows to select the maximum fitness value for each classifier and experiment
  filtered_acc <- acc %>%
    group_by(classifierName, experiment) %>% # Group by classifier and experiment
    slice_max(fitnessValue, with_ties = FALSE) %>%  # Get the row with the highest fitness for each group
    ungroup()
  filtered_acc <- as.data.frame(filtered_acc)
  
  filtered_acc$FSMethod <- FSMethod
  
  result <- filtered_acc[,c('experiment','dataset','classifierName','FSMethod','numOfSelectedFea','accuracy')]
  result$classifierName <- gsub("Neural Networks", "nn", result$classifierName)
  result$classifierName <- gsub("Random Forest", "rf", result$classifierName)
  result$classifierName <- gsub("Decision Tree", "dt", result$classifierName)
  result$classifierName <- gsub("SVM", "svm", result$classifierName)
  return(result)
}
