setwd("D:/Data science")
library(randomForest)
library(caTools)
library(smotefamily) 
library(caret)
library(ggplot2)


df_og <- read.csv("creditcard.csv")

# Data Visualisation
# Check class imbalance

cat("\n Checking Class Imbalance...")

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

plot1 <- ggplot(data = df_og, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme

print(plot1)

cat("\n Data highly imbalanced. \n SMOTE being implemented...")

#set number of fraud and legitimate cases and desired % of legitimate cases
n0 <- nrow(subset(df_og,Class==0))
n1 <- nrow(subset(df_og,Class==1))
r0 <- 0.65

#Calculate value for dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0/n1) - 1

# Create synthetic fraud cases with SMOTE
set.seed(1234)
smote_output = SMOTE(X = df_og[ , -c(1,31)], target = df_og$Class, K = 5, dup_size = ntimes)

#smote output
df_new <- smote_output$data
colnames(df_new)[30] <- "Class"

df_new$Class <- as.factor(df_new$Class)

#Data Visualization
#SMOTE output

plot2 <- ggplot(data = df_new, aes(x = factor(Class), 
                         y = prop.table(after_stat(count)), fill = factor(Class),
                         label = scales::percent(prop.table(after_stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme

print(plot2)

cat("\n Sampling data randomly....")
# sample data randomly
set.seed(333)
x <- sample(1:nrow(df_new),50000)

df <- df_new[x, ]

cat("\n Splitiing dqta into train and test")
# Splitting data in train and test data
set.seed(444)
split <- sample.split(df, SplitRatio = 0.7)
train <- subset(df, split == "TRUE")
test <- subset(df, split == "FALSE")

cat("\n Training RF model...")
trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE, verboseIter = FALSE, savePredictions = TRUE)
modfit <- train(Class ~ ., data = train, method = "rf", trControl = trControl)
testclass <- predict(modfit,test)

cat("Model trained successfully!")

cfMatrix <- confusionMatrix(testclass, as.factor(test$Class))
print(cfMatrix)


# Logistic Regression

cat("Train LR model")
set.seed(766)

reguarlized_model <- train(Class ~ ., data = train, 
                           method = "glmnet", 
                           metric = "Accuracy",
                           
                           trControl = trainControl(method = "cv", 
                                                    number = 10,
                                                    search = "random",
                                                    verboseIter = T))

cat("Model trained successfully!")
t2 <- predict(reguarlized_model,test)
cm <- confusionMatrix(t2,as.factor(test$Class))
print(cm)

output <- data.frame(metric=rep(c('Accuracy', 'Sensitivity','Specificity', 'Precision'), each=4),
                 position=rep(c('Logistic Regression', 'Random Forest'), times=2),
                 percentage=c(99.51,98.52,98.76,99.84,96.32,99.19,91.01,98.38))


plot3 <- ggplot(output, aes(fill=position, y=percentage, x=metric)) + 
  geom_bar(position='dodge', stat='identity')

print(plot3)










