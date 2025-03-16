#Imports
library(readxl)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(reshape2)
library(MASS)
library(caret)

#Read Dataset
combined_diabetes_dataset_path <- paste0('C:/Users/bowes/Downloads/combined_diabetes_dataset.xlsx')
combined_diabetes_dataset <- read_excel(paste0(combined_diabetes_dataset_path), sheet = 1)

#Maybe skip Age, FamilyHistory, DietType, Hypertension, MedicationUse, Outcome
combined_diabetes_dataset[, !(names(combined_diabetes_dataset) %in% c("Age", "FamilyHistory", "DietType", "Hypertension", "MedicationUse", "Outcome"))] <- 
  scale(combined_diabetes_dataset[, !(names(combined_diabetes_dataset) %in% c("Age", "FamilyHistory", "DietType", "Hypertension", "MedicationUse", "Outcome"))])
#Create a correlation dataset
combined_diabetes_dataset_cor <- cor(combined_diabetes_dataset, use = "complete.obs", method = "pearson")

####----Plotting----####
#Plot Correlation
corrplot(combined_diabetes_dataset_cor, 
         method = "color", 
         type = "lower", 
         tl.cex = 0.8, tl.col = "black", 
         col = colorRampPalette(c("blue", "white", "red"))(200))
#Advanced Plotting
cor_melted <- melt(combined_diabetes_dataset_cor)
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap", fill = "Correlation")

####----Regression----####
#Logistic regression model
combined_diabetes_dataset <- data.frame(combined_diabetes_dataset)
logit_model <- glm(Outcome ~ ., data = combined_diabetes_dataset, family = binomial)
summary(logit_model)

#Stepwise regression for feature selection
best_model <- stepAIC(logit_model, direction = "both")
summary(best_model)

#Predict
predicted_probs <- predict(best_model, type = "response")

#Convert probabilities to binary
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)
#Confusion Matrix
confusionMatrix(as.factor(predicted_classes), as.factor(combined_diabetes_dataset$Outcome))
