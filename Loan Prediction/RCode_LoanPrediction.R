install.packages("corrgram")
install.packages("DataCombine")
install.packages("C50")
install.packages("gridExtra")
install.packages("ggcorrplot")
install.packages("pacman")
install.packages("caret")
install.packages("e1071")
install.packages("randomForest")
install.packages("inTrees")
install.packages("corrplot")
pacman::p_load(corrgram,DataCombine,C50,gridExtra,ggplot2,caret,randomForest,inTrees,DMwR,corrplot)

rm(list=ls())
setwd(path)
getwd()
train = read.csv('train.csv')
str(train)

#######################################TrainData#######################################

#Missing Value
MissingValue = data.frame(apply(train,2,function(f){sum(is.na(f))}))
train = knnImputation(train, k = 3)

#Outiliers
NumericVariables_Index = sapply(train,is.numeric)
NumericVariables = train[,NumericVariables_Index]
ColumnNames = colnames(NumericVariables)
for (i in 1:length(ColumnNames))
{
  assign(paste0("BP_Train",i), ggplot(aes_string(y = (ColumnNames[i]), x = "Loan_Status"), 
                                           data = subset(train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=ColumnNames[i],x="Loan_Status")+
           ggtitle(paste("Box plot of Loan Status for",ColumnNames[i])))
}

gridExtra::grid.arrange(BP_Train1,BP_Train2,BP_Train3,ncol=3)
gridExtra::grid.arrange(BP_Train4,BP_Train5,ncol=2)

for(i in ColumnNames)
{
   print(i)
   val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
   #print(length(val))
   train = train[which(!train[,i] %in% val),]
}

#Correlation Analysis
corrplot(cor(NumericVariables), method = 'color', addCoef.col = "grey")

#Chi-Squared Test
FactorVaiables_Index = sapply(train,is.factor)
FactorVariables = train[,FactorVaiables_Index]
names(FactorVariables)
for (i in 1:8)
{
  print(names(FactorVariables)[i])
  print(chisq.test(table(FactorVariables$Loan_Status,FactorVariables[,i])))
}

#Dimentionality Reduction
train = subset(train,select = -c(Gender, Married, Dependents, Education, Self_Employed))

#Feature Scaling
str(train)
colNames = c('ApplicantIncome','CoapplicantIncome','LoanAmount')
for(i in colNames)
{
  print(i)
  train[,i] = (train[,i]-min(train[,i]))/
    (max(train[,i])-min(train[,i]))
}

#######################################End#######################################

#######################################TestData#######################################
test = read.csv('test.csv')
MissingValue_Test = data.frame(apply(test,2,function(f){sum(is.na(f))}))
test = knnImputation(test, k = 3)

#Correlation Analysis
NumericVariables_Index_Test = sapply(test,is.numeric)
NumericVariables_Test = test[,NumericVariables_Index_Test]
ColumnNames_Test = colnames(NumericVariables_Test)

corrplot(cor(NumericVariables_Test), method = 'color', addCoef.col = "grey")

#Dimensionality Reduction
test = subset(test,select = -c(Gender, Married, Dependents, Education, Self_Employed))

#Feature Scaling
str(test)
colNames_Test = c('ApplicantIncome','CoapplicantIncome','LoanAmount')
for(i in colNames_Test)
{
  print(i)
  test[,i] = (test[,i]-min(test[,i]))/
    (max(test[,i])-min(test[,i]))
}

#######################################End#######################################

#######################################Model Development#######################################
#Splitting the data into train and validation
Index = createDataPartition(train$Loan_Status, p = .80, list = FALSE)
TrainData = train[Index,]
ValidationData = train[-Index,]

##Random Forest
TrainData$Loan_ID = as.numeric(TrainData$Loan_ID)
ValidationData$Loan_ID = as.numeric(ValidationData$Loan_ID)
RF_Model = randomForest(Loan_Status ~ .,TrainData, importance = TRUE, ntree = 500)
List = RF2List(RF_Model)
Rules = extractRules(List, TrainData[,-8])
readableRules = presentRules(Rules, colnames(TrainData))
Metrics = getRuleMetric(Rules, TrainData[,-8], TrainData$Loan_Status)
RF_Model_Pred = predict(RF_Model,ValidationData[,-8])

#Performance Evaluation
ConfMatrix_Table_RF = table(ValidationData$Loan_Status, RF_Model_Pred)
confusionMatrix(ConfMatrix_Table_RF)
#Accuracy = 84.9%

##Decision Tree
DecisionTree_C50 = C5.0(Loan_Status ~.,TrainData,trails = 500, rules = TRUE)
summary(DecisionTree_C50)
DT_Model_Pred = predict(DecisionTree_C50, ValidationData[,-8], type = "class")

#Performance Evaluation
ConfMatrix_Table_Dtree = table(ValidationData$Loan_Status, DT_Model_Pred)
confusionMatrix(ConfMatrix_Table_Dtree)
#Accuracy = 82.19

#Applying RF to Test Data

train$Loan_ID = as.numeric(train$Loan_ID)
test$Loan_ID = as.numeric(test$Loan_ID)
RF_Model = randomForest(Loan_Status ~ .,train, importance = TRUE, ntree = 500)
List = RF2List(RF_Model)
Rules = extractRules(List, train[,-8])
readableRules = presentRules(Rules, colnames(train))
Metrics = getRuleMetric(Rules, train[,-8], train$Loan_Status)
RF_Model_Pred = data.frame(predict(RF_Model,test[,-8]))
write.csv(RF_Model_Pred,'RandomForestModel.csv',row.names = F)

