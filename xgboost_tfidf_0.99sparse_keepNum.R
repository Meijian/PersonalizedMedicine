library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(stringr)
library(tm)
library(syuzhet) 

# LabelCount Encoding function
labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}

# Load CSV files
cat("Read data")

train_text <- do.call(rbind,strsplit(readLines('C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\training_text.txt'),'||',fixed=T))
train_text <- as.data.table(train_text)
train_text <- train_text[-1,]
colnames(train_text) <- c("ID", "Text")
train_text$ID <- as.numeric(train_text$ID)

test_text <- do.call(rbind,strsplit(readLines('C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\test_text.txt'),'||',fixed=T))
test_text <- as.data.table(test_text)
test_text <- test_text[-1,]
colnames(test_text) <- c("ID", "Text")
test_text$ID <- as.numeric(test_text$ID)

train <- fread("C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\training_variants.csv", sep=",", stringsAsFactors = T)
test <- fread("C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\test_variants.csv", sep=",", stringsAsFactors = T)
trainAnno=fread("C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\train_variant_annotation_0717_tfidf_mutLoc.csv", sep=",", stringsAsFactors = T)
testAnno=fread("C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\test_variant_annotation_0717_tfidf_mutLoc.csv", sep=",", stringsAsFactors = T)
dataAnno=rbind(trainAnno,testAnno)
dataAnno[dataAnno=='']=NA
dataAnno[dataAnno=='.']=NA
dataAnno1=data.frame(dataAnno)


train <- merge(train,train_text,by="ID")
test <- merge(test,test_text,by="ID")
rm(test_text,train_text);gc()

test$Class <- -1
data <- rbind(train,test)
rm(train,test);gc()

# Basic text features
cat("Basic text features")
data$nchar <- as.numeric(nchar(data$Text))
data$nwords <- as.numeric(str_count(data$Text, "\\S+"))

# TF-IDF
cat("TF-IDF")
txt <- Corpus(VectorSource(data$Text))
txt <- tm_map(txt, stripWhitespace)
txt <- tm_map(txt, content_transformer(tolower))
txt <- tm_map(txt, removePunctuation)
txt <- tm_map(txt, removeWords, stopwords("english"))
txt <- tm_map(txt, stemDocument, language="english")
txt <- tm_map(txt, removeNumbers)
dtm <- DocumentTermMatrix(txt, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.99)
data <- cbind(data, as.matrix(dtm))

# LabelCount Encoding for Gene and Variation
# We can do more advanced feature engineering later, e.g. char-level n-grams


dataAnno1$Oncogenicity=labelCountEncoding(dataAnno1$Oncogenicity)
dataAnno1$Ref=labelCountEncoding(dataAnno1$Ref)
dataAnno1$Alt=labelCountEncoding(dataAnno1$Alt)
dataAnno1$RefAlt=labelCountEncoding(dataAnno1$RefAlt)
dataAnno1$Func.refgene=labelCountEncoding(dataAnno1$Func.refgene)
dataAnno1$SIFT_pred=labelCountEncoding(dataAnno1$SIFT_pred)
dataAnno1$Polyphen2_HDIV_pred=labelCountEncoding(dataAnno1$Polyphen2_HDIV_pred)
dataAnno1$Polyphen2_HVAR_pred=labelCountEncoding(dataAnno1$Polyphen2_HVAR_pred)
dataAnno1$LRT_pred=labelCountEncoding(dataAnno1$LRT_pred)
dataAnno1$MutationTaster_pred=labelCountEncoding(dataAnno1$MutationTaster_pred)
dataAnno1$MutationAssessor_pred=labelCountEncoding(dataAnno1$MutationAssessor_pred)
dataAnno1$FATHMM_pred=labelCountEncoding(dataAnno1$FATHMM_pred)
dataAnno1$RadialSVM_pred=labelCountEncoding(dataAnno1$RadialSVM_pred)
dataAnno1$LR_pred=labelCountEncoding(dataAnno1$LR_pred)
dataAnno1$SF=labelCountEncoding(dataAnno1$SF)
dataAnno1$proteinClass1=labelCountEncoding(dataAnno1$proteinClass1)
dataAnno1$proteinClass2=labelCountEncoding(dataAnno1$proteinClass2)
dataAnno1$Somatic=labelCountEncoding(dataAnno1$Somatic)
dataAnno1$Germline=labelCountEncoding(dataAnno1$Germline)
dataAnno1$Tissue.Type=labelCountEncoding(dataAnno1$Tissue.Type)
dataAnno1$Molecular.Genetics=labelCountEncoding(dataAnno1$Molecular.Genetics)
dataAnno1$Role.in.Cancer=labelCountEncoding(dataAnno1$Role.in.Cancer)
dataAnno1$Translocation.Partner=labelCountEncoding(dataAnno1$Translocation.Partner)
dataAnno1$Other.Germline.Mut=labelCountEncoding(dataAnno1$Other.Germline.Mut)
dataAnno1$exon_Num=labelCountEncoding(dataAnno1$exon_Num)
dataAnno1$ExonicFunc.refgene=labelCountEncoding(dataAnno1$ExonicFunc.refgene)
dataAnno=data.table(dataAnno1)


# Set seed
set.seed(2016)
cvFoldsList <- createFolds(data$Class[data$Class > -1], k=5, list=TRUE, returnTrain=FALSE) #replace it with Russ's fold2
cvfoldstr=fread("C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\folds.csv", sep=",", stringsAsFactors = T)

fold1=list(Fold1=cvfoldstr[fivefold1==1,which=TRUE],Fold2=cvfoldstr[fivefold1==2,which=TRUE],Fold3=cvfoldstr[fivefold1==3,which=TRUE],Fold4=cvfoldstr[fivefold1==4,which=TRUE],Fold5=cvfoldstr[fivefold1==5,which=TRUE])
fold2=list(Fold1=cvfoldstr[fivefold2==1,which=TRUE],Fold2=cvfoldstr[fivefold2==2,which=TRUE],Fold3=cvfoldstr[fivefold2==3,which=TRUE],Fold4=cvfoldstr[fivefold2==4,which=TRUE],Fold5=cvfoldstr[fivefold2==5,which=TRUE])

# Sentiment analysis
data$Gene <- labelCountEncoding(data$Gene)
data$Variation <- labelCountEncoding(data$Variation)
cat("Sentiment analysis")
sentiment <- get_nrc_sentiment(data$Text) 
data <- cbind(data,sentiment) 
data2=cbind(data,dataAnno)
# To sparse matrix
cat("Create sparse matrix")
#varnames <- setdiff(colnames(data), c("ID", "Class", "Text"))
varnames2 <- setdiff(colnames(data2), c("ID", "Class", "Text"))

#train_sparse <- Matrix(as.matrix(sapply(data[Class > -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
#test_sparse <- Matrix(as.matrix(sapply(data[Class == -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)

train_sparse <- Matrix(as.matrix(sapply(data2[Class > -1, varnames2, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data2[Class == -1, varnames2, with=FALSE],as.numeric)), sparse=TRUE)


y_train <- data[Class > -1,Class]-1
test_ids <- data[Class == -1,ID]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)

# Params for xgboost
param <- list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              eta = .2,
              gamma = 1,
              max_depth = 5,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

# Cross validation - determine CV scores & optimal amount of rounds
cat("XGB cross validation")
xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1000,
                 maximize = FALSE,
                 prediction = TRUE,
                 #folds = cvFoldsList,
                 folds = fold2,
                 print_every_n = 5,
                 early_stop_round = 100
)
rounds <- which.min(xgb_cv[[4]]$test_mlogloss_mean)

# Train model
cat("XGB training")
xgb_model <- xgb.train(data = dtrain,
                       params = param,
                       watchlist = list(train = dtrain),
                       nrounds = rounds,
                       verbose = 1,
                       print_every_n = 5
)

# Feature importance
cat("Plotting feature importance")
names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names,model=xgb_model)
xgb.plot.importance(importance_matrix[1:30,],1)

# Predict and output csv
cat("Predictions")
preds <- as.data.table(t(matrix(predict(xgb_model, dtest), nrow=9, ncol=nrow(dtest))))
colnames(preds) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.table(data.table(ID=test_ids, preds), "C:\\Users\\meguan\\Desktop\\Master\\Kaggle\\submission_Anno_keepNum_sparse0.99_fold2.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)