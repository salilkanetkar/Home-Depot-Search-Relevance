penloss=function(par,X,S,Y,lambda){
  Pword=dim(X)[2]
  Psearch=dim(S)[2]    
  Uhat=matrix(par, nrow=Pword, ncol=Psearch)
  Yhat=diag((X%*%Uhat)%*%t(S))
  loss=sum((Y-Yhat)^2)
  penalty=lambda*sum(sum(par^2))
  return(loss)
}

lambda = 0.1
Pword = dim(svd_title_description[1:1000,])[2]
Psearch = dim(svd_search[1:1000,])[2]  
opt.out=optim(par = rep(0,Psearch*Pword), method="Nelder-Mead",fn = penloss, X=as.matrix(svd_title_description[1:1000,]), S=as.matrix(svd_search[1:1000,]), Y=as.matrix(scale(Y_train[1:1000,1])), lambda=lambda)
Uhat.opt=matrix(opt.out$par,nrow=Pword, ncol=Psearch)
yhat.opt=diag(as.matrix(svd_title_description[1:1000,])%*%Uhat.opt%*%t(as.matrix(svd_search[1:1000,])))
cor(Y_train[1:1000,1],yhat.opt)^2
yhat.opt_test=diag(as.matrix(test_svd_title_description[1:1000,])%*%Uhat.opt%*%t(as.matrix(test_svd_search[1:1000,])))
