# drawPearsonResult:This function visualizes the results of feature selection based on Pearson Correlation Coefficient (PCC). 
# It shows the number of selected features at various correlation thresholds with a bar chart, 
# and overlays the classification accuracy (by classifier type) to illustrate the relationship between accuracy and the number of selected features.
drawPearsonResult <- function(ave,totalFeature){
  maxFeature = max(ave$argNumOfSelectedFea)
  min = min(ave$argAccuracy)
  max = max(ave$argAccuracy)
  scaling = (maxFeature-5)/(max-min)
  ggplot(data = ave,
         mapping = aes(
           x = cutoff))+
    ylim(c(0,2*maxFeature))+
    geom_bar(aes(x = cutoff,y=argNumOfSelectedFea/4),stat = "identity", fill = "#984ea3")+
    geom_text(aes(y=argNumOfSelectedFea,label=round(argNumOfSelectedFea,0)),vjust = -0.5,size = 3,color = "#984ea3")+
    geom_point(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5), fill = classifierName,shape = classifierName,color = classifierName))+
    geom_line(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5), color = classifierName) )+
    geom_text(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5),label= paste(round(argAccuracy*100,2),"%") , color = classifierName),vjust = -0.5,size = 3)+
    scale_color_manual(name="Accuracy Gained By Classifier",values = c("SVM" ="#ff7f00",
                                                                       "Neural Networks" ="#4daf4a",
                                                                       "Decision Tree" ="#377eb8",
                                                                       "Random Forest" ="#e41a1c"))+
    scale_fill_manual(name="Accuracy Gained By Classifier",values = c("SVM" ="#ff7f00",
                                                                      "Neural Networks" ="#4daf4a",
                                                                      "Decision Tree" ="#377eb8",
                                                                      "Random Forest" ="#e41a1c"))+
    scale_shape_manual(name="Accuracy Gained By Classifier",values = c("SVM" =15,
                                                                       "Neural Networks" =16,
                                                                       "Decision Tree" =17,
                                                                       "Random Forest" =18))+
    ylab("The average number of selected features")+
    labs(title = "Pearson Correlation Coefficient-Based Feature Selection Result" ,tag = ave$dataset)+
    theme_classic()+
    theme(
      plot.caption = element_text(face = "bold"),
      axis.title.y = element_text(face = "bold"),
      axis.title.x = element_text(face = "bold"),
      legend.position = "top",
      title = element_text(face = "bold"),
      axis.text = element_text(size = 8),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.tag = element_text(size = 8,face = "bold.italic"),
      plot.tag.position = "topright"
    )
}

# drawMutualInfoResult:This function visualizes the results of feature selection based on Mutual Information (MI). 
# It uses a bar chart to display the number of features selected at various MI boundaries 
# and combines it with accuracy data to show the relationship between the number of features selected and classification accuracy.
drawMutualInfoResult <- function(ave,totalFeature){
  ave$orderOfBoundary = rep(seq(6,1,-1),4)
  maxFeature = max(ave$argNumOfSelectedFea)
  min = min(ave$argAccuracy)
  max = max(ave$argAccuracy)
  scaling = (maxFeature-5)/(max-min)
  # Xlab = rev(c(0.01,0.10,0.20,0.30,0.40,0.50))
  Xlab = rev(c(unique(ave$boundary)))
  ggplot(data = ave,
         mapping = aes(
           x = orderOfBoundary))+
    ylim(c(0,2*maxFeature))+
    geom_bar(aes(x = orderOfBoundary,y=argNumOfSelectedFea/4),stat = "identity", fill = "#984ea3")+
    geom_text(aes(y=argNumOfSelectedFea,label=round(argNumOfSelectedFea,0)),vjust = -0.5,size = 3,color = "#984ea3")+
    geom_point(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5), fill = classifierName,shape = classifierName,color = classifierName))+
    geom_line(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5), color = classifierName) )+
    geom_text(mapping = aes(y = (argAccuracy-min)*scaling+(maxFeature+5),label= paste(round(argAccuracy*100,2),"%") , color = classifierName),vjust = -0.5,size = 3)+
    scale_color_manual(name="Accuracy Gained By Classifier",values = c("SVM" ="#ff7f00",
                                                                       "Neural Networks" ="#4daf4a",
                                                                       "Decision Tree" ="#377eb8",
                                                                       "Random Forest" ="#e41a1c"))+
    scale_fill_manual(name="Accuracy Gained By Classifier",values = c("SVM" ="#ff7f00",
                                                                      "Neural Networks" ="#4daf4a",
                                                                      "Decision Tree" ="#377eb8",
                                                                      "Random Forest" ="#e41a1c"))+
    scale_shape_manual(name="Accuracy Gained By Classifier",values = c("SVM" =15,
                                                                       "Neural Networks" =16,
                                                                       "Decision Tree" =17,
                                                                       "Random Forest" =18))+
    xlab("The Mutual Information Boundary of selected features")+
    ylab("The average number of selected features")+
    scale_x_continuous(breaks = seq(1,6,1), labels = Xlab)+
    labs(title ="Mutual Information-Based Feature Selection Result" , tag = ave$dataset)+
    theme_classic()+
    theme(
      plot.caption = element_text(face = "bold"),
      axis.title.y = element_text(face = "bold"),
      axis.title.x = element_text(face = "bold"),
      legend.position = "top",
      title = element_text(face = "bold"),
      axis.text = element_text(size = 8),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.tag = element_text(size = 8,face = "bold.italic"),
      plot.tag.position = "topright"
    )
}

#drawResult: This function visualizes the feature selection and classification result in each repeat experiment. 
#It plots a bar chart and line graph to show the number of selected features and their corresponding accuracy. 
#The x-axis represents the experiment number, and the y-axis shows the number of features selected, with accuracy values annotated.
drawResult <- function(result,accuracy_allFeatures,algorithm,dataset,totalFeature){
  numOfFeature = length(result[[1]]$alphaFeatSetIndex)
  sumResult <- data.frame(
    matrix(data = NA,
           ncol = 5,
           nrow = 10,
           dimnames = list(c(1:10),
                           c("orderOfNumOfFeatures","Number","originalFeatSetAcc","bGWOSelectedFeatSetAcc","numOfFeatures")))
  )
  for(i in 1:length(result)){
    sumResult$Number[i] = i
    sumResult$numOfFeatures[i] = result[[i]]$numAlphaFeature
    sumResult$originalFeatSetAcc[i] = 1- as.numeric(accuracy_allFeatures[[i]])
    sumResult$bGWOSelectedFeatSetAcc[i] = 1- result[[i]]$smallestError
   }
  sumResult = sumResult[order(sumResult$numOfFeatures),]
  sumResult$orderOfNumOfFeatures = c(1:10)
  min = 0.90
  max = 1
  scaling = 1/3*totalFeature/(max-min)
  Xlab = c(sumResult$Number)
  ggplot(sumResult)+
    geom_bar(aes(x=orderOfNumOfFeatures,y=numOfFeatures),stat="identity",fill = "#984ea3")+
    ylim(c(0,3/2*totalFeature))+
    geom_text(aes(x=orderOfNumOfFeatures,y=numOfFeatures,label=numOfFeatures),vjust=-0.5,size=3,color = "#984ea3")+
    geom_line(aes(x=orderOfNumOfFeatures,y=(originalFeatSetAcc-min)*scaling+numOfFeature,col = "originalFeatSetAcc"))+
    geom_point(aes(x=orderOfNumOfFeatures,y=(originalFeatSetAcc-min)*scaling+numOfFeature,color = "originalFeatSetAcc",shape="originalFeatSetAcc",fill="originalFeatSetAcc"),size=2)+
    geom_line(aes(x=orderOfNumOfFeatures,y=(bGWOSelectedFeatSetAcc-min)*scaling+numOfFeature,col = "bGWOSelectedFeatSetAcc"))+
    geom_point(aes(x=orderOfNumOfFeatures,y=(bGWOSelectedFeatSetAcc-min)*scaling+numOfFeature,color = "bGWOSelectedFeatSetAcc",shape="bGWOSelectedFeatSetAcc",fill="bGWOSelectedFeatSetAcc"),size=2)+
    geom_text(aes(x=orderOfNumOfFeatures,y=(originalFeatSetAcc-min)*scaling+numOfFeature,label=paste(round(originalFeatSetAcc*100,2),"%"),color="originalFeatSetAcc"),size=2.5,vjust = +2,check_overlap = TRUE)+
    geom_text(aes(x=orderOfNumOfFeatures,y=(bGWOSelectedFeatSetAcc-min)*scaling+numOfFeature,label=paste(round(bGWOSelectedFeatSetAcc*100,2),"%"),color = "bGWOSelectedFeatSetAcc"),size=2.5,vjust = -2,check_overlap = TRUE)+
    # scale_x_continuous(breaks = seq(1,10,1))+
    xlab("The serial number of experiment")+
    ylab("The number of features selected")+
    scale_x_continuous(breaks = seq(1,10,1), labels = Xlab)+
    labs(title = algorithm,tag = dataset)+
    scale_color_manual(name="Accuracy",values = c("originalFeatSetAcc"="#4daf4a","bGWOSelectedFeatSetAcc"="#ff7f00"))+
    scale_fill_manual(name="Accuracy",values = c("originalFeatSetAcc"="#4daf4a","bGWOSelectedFeatSetAcc"="#ff7f00"))+
    scale_shape_manual(name="Accuracy",values = c("originalFeatSetAcc"=21,"bGWOSelectedFeatSetAcc"=22))+
    theme_classic()+
    theme(
      plot.caption = element_text(face = "bold"),
      axis.title.y = element_text(face = "bold"),
      axis.title.x = element_text(face = "bold"),
      legend.position = "top",
      title = element_text(face = "bold"),
      axis.text = element_text(size = 8),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.tag = element_text(size = 8,face = "bold.italic"),
      plot.tag.position = "topright"
    )
  
}

# drawResultSumNew:This function summarizes and visualizes all experimental results,
# displaying the number of features selected and classification accuracy for different feature selection methods. 
# It generates a boxplot for each classifier and feature selection method, 
# showing the distribution of accuracy and the number of features selected.
drawResultSumNew <- function( result ,original_iotD){
  result$classifierName <- gsub("svm", "SVM", result$classifierName)
  result$classifierName <- gsub("nn", "Neural Networks", result$classifierName)
  result$classifierName <- gsub("dt", "Decision Tree", result$classifierName)
  result$classifierName <- gsub("rf", "Random Forest", result$classifierName)
  result$classifierName <- factor(result$classifierName,levels = c("SVM","Neural Networks","Decision Tree","Random Forest",ordered = TRUE))
  result$FSMethod <- factor(result$FSMethod,levels = c("ORIGINAL","PCC","MI","BGA","BPSO","BGWO",ordered = TRUE))
  result_gathered <- gather(result, key = "Metrics",value = "Value"
                            ,-experiment
                            ,-dataset
                            ,-classifierName
                            ,-FSMethod)
  result_gathered$Metrics <- gsub("numOfSelectedFea", "Number of Selected Features", result_gathered$Metrics)
  result_gathered$Metrics <- gsub("accuracy", "Accuracy", result_gathered$Metrics)
  
  result_gathered <- as.data.frame(result_gathered)

  ggplot(result_gathered) +
    aes(x = FSMethod, y = Value, fill = FSMethod) +
    geom_boxplot() +
    facet_grid(
      vars(Metrics),
      vars("Classifier Name"),
      scales = "free_y"
    )+
    scale_fill_manual(
      name="Feature Selection Method:",
      values = 
        c(ORIGINAL = "#440154", 
          PCC = "#404385",
          MI = "#2A778E",
          BGA = "#27A882",
          BPSO = "#7BD04F",
          BGWO = "#FDE725")
      
    ) +
    labs(
      x = "Feature Selection Methods",
      y = "Value",
      tag = unique(result_gathered$dataset)
    ) +
    theme_classic()+
    theme(
      plot.caption = element_text(face = "bold"),
      axis.title.y = element_text(face = "bold"),
      axis.title.x = element_text(face = "bold"),
      axis.text.x = element_text(size = 8,face = "bold"),
      legend.position = "top",
      title = element_text(face = "bold"),
      axis.text = element_text(size = 10),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      plot.tag = element_text(size = 12,face = "bold.italic"),
      plot.tag.position = "topright",
      strip.text = element_text(size = 12,face="bold")
    )+
    facet_grid(
      vars(Metrics),
      vars(classifierName),
      scales = "free_y"
    )
}

# drawConfusionMatrix: This function visualizes the confusion matrix of a classifier, 
# which compares the predicted labels to the real labels. 
# The matrix uses color gradients and numeric labels to provide an intuitive view of the classifier's performance.
drawConfusionMatrix <- function(p_rf,dataset){
  p_rf <- as.data.frame(p_rf)
  ggplot(p_rf) +
    aes(x = Var2, y = p, fill = Freq) +
    geom_tile(size = 1.2) +
    geom_text(aes(label =Freq))+
    scale_fill_gradient(low ='#f7fcfd' , high ='#016c59' ) +
    labs(title = "Confusion Matrix",tag = dataset)+
    xlab(label = "Real Labels")+
    ylab(label = "Predicted Result")+
    theme_classic() +
    theme(
      plot.caption = element_text(face = "bold"),
      axis.title.y = element_text(face = "bold"),
      axis.title.x = element_text(face = "bold"),
      legend.position = "top",
      title = element_text(face = "bold"),
      axis.text = element_text(size = 10),
      axis.text.x = element_text(angle = 30,hjust = 1,vjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 9),
      plot.tag = element_text(size = 8,face = "bold.italic"),
      plot.tag.position = "topright"
    )
}