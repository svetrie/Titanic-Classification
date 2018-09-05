library(magrittr)
library(dplyr)
library(stringr)
library(ggplot2)

training_set = read.csv('train.csv')
test_set = read.csv('test.csv')
test_set$Survived = rep(NA, nrow(test_set))

titanic_data = rbind(training_set, test_set)
titanic_data$Survived = as.factor(titanic_data$Survived)

#EXPLORATORY DATA ANALYSIS

#Visualize survival rates for each gender
ggplot(titanic_data[!is.na(titanic_data$Survived), ], aes(Sex, fill = Survived)) +
  geom_bar()

#Visualize survival rates for different passenger classes
ggplot(titanic_data[!is.na(titanic_data$Survived), ], aes(Pclass, fill = Survived)) +
  geom_bar()

#Compare survial rates for different embarked locations
table(titanic_data$Survived, titanic_data$Embarked)

#FEATURE ENGINEERING 

#Create a Title category by extracting the titles from people's names 
titanic_data$Title = str_extract(titanic_data$Name, "[A-Z][[:alpha:]]+\\.")
count(titanic_data, Title)
titanic_data$Title = ifelse(titanic_data$Title %in% c("Capt.", "Col.", "Major.","Countess.", "Don.", "Dona.", "Jonkheer.", "Lady.", "Sir.", "Dr."), "UpperClass",
                        ifelse(titanic_data$Title == "Mlle.", "Miss.",
                            ifelse(titanic_data$Title %in% c("Mme.", "Ms."), "Mrs.", titanic_data$Title)))

#Visualize survival rates for different titles
ggplot(titanic_data[!is.na(titanic_data$Survived), ], aes(Title, fill = Survived)) +
  geom_bar()

#Create a family size category by adding SibSp and Parch
titanic_data$FamSize = titanic_data$SibSp + titanic_data$Parch

#Visualize survival rates for different family sizes
ggplot() +
  geom_bar(data = titanic_data[!is.na(titanic_data$Survived), ], aes(FamSize, fill = Survived))


#REPLACE MISSING VALUES

#Replace missing Embarked locations
titanic_data[titanic_data$Embarked == "", ]
table(titanic_data$Embarked, titanic_data$Pclass) #Most common embarked location for first class passengers is 'C'
titanic_data[titanic_data$Embarked == "", ]$Embarked = "C"

#Replace missing Fare prices
titanic_data[is.na(titanic_data$Fare), ]
#Replace this passenger's fare with the median fare price for his passenger class and embarked location
titanic_data[is.na(titanic_data$Fare), ]$Fare = median(titanic_data[titanic_data$Pclass == 3 & titanic_data$Embarked == "S", ]$Fare, na.rm = TRUE) 

#Encode factors and apply feature scaling
titanic_data$Embarked = factor(titanic_data$Embarked, levels(titanic_data$Embarked), labels = c(0,1,2,3))
titanic_data$Title = factor(titanic_data$Title, levels = unique(titanic_data$Title), labels = c(0,1,2,3,4,5))
titanic_data$Pclass = factor(titanic_data$Pclass, levels = c(1,2,3), labels = c(1,2,3))

titanic_data[, c(6, 7, 8, 10, 14)] = scale(titanic_data[, c(6, 7, 8, 10, 14)])

#Create a predictive model for Age to replace missing Age values. Compare models using rmse
library(Metrics)
library(e1071)
svm_regressor = svm(formula = Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title,
                data = titanic_data[!is.na(titanic_data$Age), ],
                type = "eps-regression",
                kernel = 'radial')

svm_pred_ages = predict(svm_regressor, newdata = titanic_data[!is.na(titanic_data$Age), ])
rmse(titanic_data[!is.na(titanic_data$Age), ]$Age, svm_pred_ages)

library(randomForest)
set.seed(1234)
randfor_regressor = randomForest(x = titanic_data[!is.na(titanic_data$Age), c(3, 5, 7, 8, 10, 12:14)],
                                 y = titanic_data$Age[!is.na(titanic_data$Age)],
                                 ntree = 100)

randfor_pred_ages = predict(randfor_regressor, titanic_data[!is.na(titanic_data$Age), c(3, 5, 7, 8, 10, 12:14)])
rmse(titanic_data[!is.na(titanic_data$Age), ]$Age, randfor_pred_ages)

titanic_data[is.na(titanic_data$Age), ]$Age = predict(randfor_regressor, titanic_data[is.na(titanic_data$Age), c(3, 5, 7, 8, 10, 12:14)])

#SURVIVAL CLASSIFICATION
training_set = titanic_data[!is.na(titanic_data$Survived), ]
test_set = titanic_data[is.na(titanic_data$Survived), ]

#Random Forest model (could potentially be overfit)
set.seed(1234)
randfor_classifier = randomForest(x = training_set[, c(3, 5:8, 10, 12:14)],
                                  y = training_set$Survived,
                                  ntree = 100)

randfor_survivals = predict(randfor_classifier, test_set[, c(3, 5:8, 10, 12:14)])

results = data.frame(test_set$PassengerId, randfor_survivals)
colnames(results) = c("PassengerID", "Survived")

write.csv(results, file = "TitanicSurvivals.csv")
