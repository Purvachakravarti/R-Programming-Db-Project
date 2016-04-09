#import libraries
library(timeDate)
library(randomForest)


#set options to make sure scientific notation is disabled when writing files
options(scipen=500)


#read in data
dfStore <- read.csv(file='~/project/data/samplestores.csv')
dfTrain <- read.csv(file='~/project/data/training.csv')
dfTest <- read.csv(file='~/project/data/testing.csv')
dfFeatures <- read.csv(file='~/project/data/storefeatures.csv')
submission = read.csv(file='~/project/data/sampleoutput.csv',header=TRUE,as.is=TRUE)


# Merge Type and Size
dfTrainTmp <- merge(x=dfTrain, y=dfStore, all.x=TRUE)
dfTestTmp <- merge(x=dfTest, y=dfStore, all.x=TRUE)


# Merge all the features
train <- merge(x=dfTrainTmp, y=dfFeatures, all.x=TRUE)
test <- merge(x=dfTestTmp, y=dfFeatures, all.x=TRUE)


# Make features for train
train$year = as.numeric(substr(train$Date,1,4))
train$month = as.numeric(substr(train$Date,6,7))
train$day = as.numeric(substr(train$Date,9,10))
train$days = (train$month-1)*30 + train$day
train$Type = as.character(train$Type)
train$Type[train$Type=="A"]=1
train$Type[train$Type=="B"]=2
train$Type[train$Type=="C"]=3
train$IsHoliday[train$IsHoliday=="TRUE"]=1
train$IsHoliday[train$IsHoliday=="FALSE"]=0
train$dayHoliday = train$IsHoliday*train$days
train$logsales = log(4990+train$Weekly_Sales)
#weight certain features more by duplication
train$tDays = 360*(train$year-2010) + (train$month-1)*30 + train$day
train$days30 = (train$month-1)*30 + train$day


#Make features for test
test$year = as.numeric(substr(test$Date,1,4))
test$month = as.numeric(substr(test$Date,6,7))
test$day = as.numeric(substr(test$Date,9,10))
test$days = (test$month-1)*30 + test$day
test$Type = as.character(test$Type)
test$Type[test$Type=="A"]=1
test$Type[test$Type=="B"]=2
test$Type[test$Type=="C"]=3
test$IsHoliday[test$IsHoliday=="TRUE"]=1
test$IsHoliday[test$IsHoliday=="FALSE"]=0
test$dayHoliday = test$IsHoliday*test$days
test$tDays = 360*(test$year-2010) + (test$month-1)*30 + test$day
test$days30 = (test$month-1)*30 + test$day


#Run model
tmpR0 = nrow(submission)
j=1
while (j < tmpR0) {
  print(j/tmpR0)#keep track of progress
  #select only relevant data for the store and department tuple
  tmpId = submission$Id[j]
  tmpStr = unlist(strsplit(tmpId,"_"))
  tmpStore = tmpStr[1]
  tmpDept = tmpStr[2]
  dataF1 = train[train$Dept==tmpDept,]
  tmpL = nrow(dataF1[dataF1$Store==tmpStore,])
  #since MAE is weighted, increase weights of holiday data by 5x
  tmpF = dataF1[dataF1$IsHoliday==1,]
  dataF1 = rbind(dataF1,do.call("rbind", replicate(4, tmpF, simplify = FALSE)))
  dataF2 = dataF1[dataF1$Store==tmpStore,]  
  testF1 = test[test$Dept==tmpDept,]
  testF1 = testF1[testF1$Store==tmpStore,]
  testRows = nrow(testF1)
  #this model is trained on store+dept filtered data
  tmpModel =  randomForest(logsales ~ year + month + day + days + dayHoliday + tDays + days30, 
                               ntree=200, replace=TRUE, mtry=3, data=dataF2)
  tmpP = exp(predict(tmpModel,testF1))-4990
  k = j + testRows - 1
  submission$Weekly_Sales[j:k] = tmpP
  j = k+1
}

#write the output to csv
write.table(x=submission,
            file='~/project/final/Final.csv',
            sep=',', row.names=FALSE, quote=FALSE)

