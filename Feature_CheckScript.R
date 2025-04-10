#Imports
library(readxl)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(reshape2)
library(MASS)
library(caret)

#Read Dataset
data_path_2 <-paste0('PGCERT/COM747_Project/numerical_stroke_data.csv')
data_path_3 <-paste0('PGCERT/COM747_Project/numerical_heart_data.csv')
dataset <- read.csv(paste0(data_path_3))

#Scale nonbinary data
#Stroke Data
dataset[, names(dataset) %in% c("Age", "RestingBP",	"Cholesterol", "MaxHR",	"Oldpeak")] <- 
  scale(dataset[, !(names(dataset) %in% c("Age", "RestingBP",	"Cholesterol", "MaxHR",	"Oldpeak"))])

#Heart Data
#dataset[, !(names(dataset) %in% c("age", "avg_glucose_level", "bmi"))] <- 
#  scale(dataset[, !(names(dataset) %in% c("age", "avg_glucose_level", "bmi"))])


#Create a correlation dataset
dataset_cor <- cor(dataset, use = "complete.obs", method = "pearson")

####----Plotting----####
corrplot(dataset_cor, 
         method = "color", 
         type = "lower", 
         tl.cex = 0.8, tl.col = "black", 
         col = colorRampPalette(c("blue", "white", "red"))(200))

cor_melted <- melt(dataset_cor)
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap", fill = "Correlation")

####----Regression----####
dataset <- data.frame(dataset)

#logit_model <- glm(Outcome ~ ., data = dataset, family = binomial)
#logit_model <- glm( ~ ., data = dataset, family = binomial)
logit_model <- glm(HeartDisease ~ ., data = dataset, family = binomial)

summary(logit_model)

#Stepwise regression for feature selection
best_model <- stepAIC(logit_model, direction = "both")
summary(best_model)
