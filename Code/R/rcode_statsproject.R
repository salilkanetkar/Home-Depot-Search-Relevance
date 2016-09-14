library(e1071)
X_train = cbind(X_train_1,Y_train)

#SVM Using Feature Set 1
X_train_small = X_train_1[1:15000,]
Y_train_small = Y_train[1:15000,]
svm_model_1 <- svm(Y_train_small ~.,data =  X_train_small ,kernel = 'radial')
pred <- predict(svm_model_1,X_test_1)
rmse_svm_radial_1 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))

#SVM Using Feature Set 2
X_train_small = X_train_2[1:10000,]
Y_train_small = Y_train[1:10000,]
svm_model_radial_2 <- svm(Y_train_small ~.,data =  X_train_small ,kernel = 'radial')
pred <- predict(svm_model_radial_2,X_test_2)
rmse_svm_radial_2 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))

#SVM Using Feature Set 3
X_train_3 <- na.omit(X_train_3)
X_test_3 <- na.omit(X_test_3)
X_train_small = X_train_3[1:10000,]
svm_model_radial_3 <- svm(V118 ~.,data =  X_train_small ,kernel = 'radial')
pred <- predict(svm_model_radial_3,X_test_3[,-118])
rmse_svm_radial_3 <- sqrt(sum((X_test_3[,118]-pred)^2)/nrow(X_test_3))

#SVM Using Feature Set 4
X_train_4 <- na.omit(X_train_4)
X_test_4 <- na.omit(X_test_4)
X_train_small = X_train_4[1:10000,]
svm_model_radial_4 <- svm(V18 ~.,data =  X_train_small ,kernel = 'radial')
pred <- predict(svm_model_radial_4,X_test_4[,-18])
rmse_svm_radial_4 <- sqrt(sum((X_test_4[,18]-pred)^2)/nrow(X_test_4))


#SVM Using Feature Set 5
X_train_small = X_train_5[1:10000,]
Y_train_small = Y_train[1:10000,]
svm_model_radial_5 <- svm(Y_train_small ~.,data =  X_train_small ,kernel = 'linear')
pred <- predict(svm_model_radial_5,X_test_5)
rmse_svm_radial_5 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))


X_train_1 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_train_1.csv", header=FALSE)
X_test_1 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_test_1.csv", header=FALSE)
X_train_2 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_train_2.csv", header=FALSE)
X_test_2 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_test_2.csv", header=FALSE)
X_train_5 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_train_5.csv", header=FALSE)
X_test_5 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_test_5.csv", header=FALSE)

Y_train <- read.table("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/Y_train.csv", quote="\"", comment.char="")
Y_test <- read.table("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/Y_test.csv", quote="\"", comment.char="")

X_train_3 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_train_3.csv", header=FALSE)

X_train_4 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_train_4.csv", header=FALSE)

X_test_3 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_test_3.csv", header=FALSE)

X_test_4 <- read.csv("H:/UCLA Winter 2016/201B - Statistical Modeling and Learning/Project/RCode/X_test_4.csv", header=FALSE)

library(randomForest)
#Random Forest Feature Set 1
rf_model_1 <- randomForest(Y_train[,1] ~.,data = X_train_1, ntree=200,mtry=10)
pred <- predict(rf_model_1,X_test_1)
rmse_rf_1 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))

#Random Forest Feature Set 2
rf_model_2 <- randomForest(Y_train[,1] ~.,data = X_train_2, ntree=200,mtry=10)
pred <- predict(rf_model_2,X_test_2)
rmse_rf_2 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))

#Random Forest Feature Set 3
rf_model_3 <- randomForest(V118 ~.,data = X_train_3, ntree=200,mtry=10)
pred <- predict(rf_model_3,X_test_3)
rmse_rf_3 <- sqrt(sum((Y_test-pred)^2)/nrow(Y_test))

#Random Forest Feature Set 4
rf_model_4 <- randomForest(V18 ~.,data = X_train_4, ntree=200,mtry=10)
pred <- predict(rf_model_4,X_test_4[,-V18])
rmse_rf_4 <- sqrt(sum((X_test_4[,V18]-pred)^2)/nrow(Y_test))

#Random Forest Feature Set 5