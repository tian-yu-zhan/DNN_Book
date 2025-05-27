
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)
library(e1071)
library(xgboost)
library(randomForest)

set.seed(1) 
n.first.itt = 10^4
n.cluster = 8

DNN.first.pbo.rate.vec = runif(n.first.itt, min = 0.1, max = 0.5)
DNN.first.delta.rate.vec = runif(n.first.itt, min = -0.1, max = 0.3)
n.stage.1 = 200
n.stage.2.1 = 100 
n.stage.2.2 = 400 
ratio.stage.2 = 1/4
theta.cutoff = 0.15

adaptive.bin.data.func = function(rate.pbo.in, rate.trt.in){
  
  rand.1.pbo = rbinom(n.stage.1/2, 1, rate.pbo.in)
  rand.1.trt = rbinom(n.stage.1/2, 1, rate.trt.in)
  rand.1.delta = mean(rand.1.trt) - mean(rand.1.pbo)
  
  adap.ind = (rand.1.delta > theta.cutoff)
  
  if (adap.ind){
    rand.2.pbo = rbinom(n.stage.2.2*ratio.stage.2, 1, rate.pbo.in)
    rand.2.trt = rbinom(n.stage.2.2*(1-ratio.stage.2), 1, rate.trt.in)
  } else {
    rand.2.pbo = rbinom(n.stage.2.1*(1-ratio.stage.2), 1, rate.pbo.in)
    rand.2.trt = rbinom(n.stage.2.1*ratio.stage.2, 1, rate.trt.in)
  }
  
  rand.2.delta = mean(rand.2.trt) - mean(rand.2.pbo)
  
  new.list = c(mean(rand.1.pbo),
               mean(rand.2.pbo),
               rand.1.delta,
               rand.2.delta,
               adap.ind
  )
  
  return(new.list)
}

cl = makeCluster(n.cluster)
registerDoParallel(cl)
DNN.first.input = foreach(first.ind=1:n.first.itt) %dopar% {

  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)
  
  set.seed(first.ind)
  
  rate.pbo = DNN.first.pbo.rate.vec[first.ind]
  rate.delta = DNN.first.delta.rate.vec[first.ind]
  rate.trt = rate.pbo + rate.delta
  
  DNN.first.train = adaptive.bin.data.func(
         rate.pbo.in = rate.pbo, 
         rate.trt.in = rate.trt)
  
  return(DNN.first.train)
  
}
stopCluster(cl)

data.train =  matrix(unlist(DNN.first.input), nrow = n.first.itt,
                        ncol = 5, byrow=TRUE)
data.train.scale =scale(data.train)
data.train.label = DNN.first.delta.rate.vec

col_means_train <- attr(data.train.scale, "scaled:center")
col_stddevs_train <- attr(data.train.scale, "scaled:scale")

set_random_seed(1)
model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'linear')

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = list('mse')
)

dnn_history = model %>% fit(
  data.train.scale,
  data.train.label,
  epochs = 500,
  batch_size = 100,
  validation_split = 0
)

print(dnn_history)

## svm
data.svm = data.frame(cbind(data.train, data.train.label))
colnames(data.svm) = c("x1", "x2", "x3", "x4", "x5", "y")
svm.fit.1 =  svm(y ~ x1+x2+x3+x4+x5, data =data.svm, scale = TRUE,
                 kernel = "linear")
svm.fit.2 =  svm(y ~ x1+x2+x3+x4+x5, data =data.svm, scale = TRUE,
                 kernel = "polynomial")
svm.fit.3 =  svm(y ~ x1+x2+x3+x4+x5, data =data.svm, scale = TRUE,
                 kernel = "radial")
svm.fit.4 =  svm(y ~ x1+x2+x3+x4+x5, data =data.svm, scale = TRUE,
                 kernel = "sigmoid")

## XGboost
xgboost.fit.1 = xgboost(data = data.train, 
                        label = data.train.label, 
                        max.depth = 6, 
                        eta = 0.3,
                        gamma = 0, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

xgboost.fit.2 = xgboost(data = data.train, 
                        label = data.train.label, 
                        max.depth = 6, 
                        eta = 0.3,
                        gamma = 1, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

xgboost.fit.3 = xgboost(data = data.train, 
                        label = data.train.label, 
                        max.depth = 6, 
                        eta = 1,
                        gamma = 0, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

xgboost.fit.4 = xgboost(data = data.train, 
                        label = data.train.label, 
                        max.depth = 6, 
                        eta = 1,
                        gamma = 1, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

### RF
rf.fit.1 = randomForest(y ~ x1+x2+x3+x4+x5, data =data.svm, 
                        ntree = 500, nodesize = 5)
rf.fit.2 = randomForest(y ~ x1+x2+x3+x4+x5, data =data.svm, 
                        ntree = 500, nodesize = 10)
rf.fit.3 = randomForest(y ~ x1+x2+x3+x4+x5, data =data.svm, 
                        ntree = 1000, nodesize = 5)
rf.fit.4 = randomForest(y ~ x1+x2+x3+x4+x5, data =data.svm, 
                        ntree = 1000, nodesize = 10)

#################################
rate.pbo.val.vec = c(0.3, 0.3, 0.4, 0.4)
rate.delta.val.vec = c(0, 0.2, 0, 0.2)
n.val.itt = 10^5

val.para.grid = data.frame("rate_pbo" = rate.pbo.val.vec,
                           "rate_delta" = rate.delta.val.vec)


val.para.grid$RF_4 = val.para.grid$RF_3 = 
  val.para.grid$RF_2 = val.para.grid$RF_1 = 
val.para.grid$XG_4 = val.para.grid$XG_3 = val.para.grid$XG_2 = 
  val.para.grid$XG_1 = 
  val.para.grid$SVM_4 = val.para.grid$SVM_3 =
  val.para.grid$SVM_2 = val.para.grid$SVM_1 =  
  
  val.para.grid$DNN = 
  val.para.grid$DNN_bias = NA
n.val.ind = dim(val.para.grid)[1]

## evaluate each scenario
for (val.ind in 1:n.val.ind){
  set.seed(val.ind)
  print(val.ind)
  
  rate.pbo.val = val.para.grid$rate_pbo[val.ind] 
  rate.delta.val = val.para.grid$rate_delta[val.ind] 
  rate.trt.val = rate.pbo.val + rate.delta.val
  
  DNN.val.fit = t(sapply(1:n.val.itt, 
                           function(temp.ind){adaptive.bin.data.func(
                             rate.pbo.in = rate.pbo.val, 
                             rate.trt.in = rate.trt.val)}))
  
  delta.1.est = DNN.val.fit[, 3]
  delta.2.est = DNN.val.fit[, 4]

  naive.1.est = 0.8*delta.1.est + 0.2*delta.2.est 
  naive.2.est = 0.5*delta.1.est + 0.5*delta.2.est
  naive.3.est = 0.2*delta.1.est + 0.8*delta.2.est
  
  DNN.test.data.scale = scale(DNN.val.fit,
                              center = col_means_train, 
                              scale = col_stddevs_train)

  ## DNN
  DNN.T1.T2.est = 
    model %>% predict(DNN.test.data.scale)
  
  ## SVM
  data.svm.val = data.frame(DNN.val.fit)
  colnames(data.svm.val) = c("x1", "x2", "x3", "x4", "x5")
  
  SVM.1.est = predict(svm.fit.1, newdata = data.svm.val)
  SVM.2.est = predict(svm.fit.2, newdata = data.svm.val)
  SVM.3.est = predict(svm.fit.3, newdata = data.svm.val)
  SVM.4.est = predict(svm.fit.4, newdata = data.svm.val)
  
  ## XG
  XG.1.est = predict(xgboost.fit.1, newdata = DNN.val.fit)
  XG.2.est = predict(xgboost.fit.2, newdata = DNN.val.fit)
  XG.3.est = predict(xgboost.fit.3, newdata = DNN.val.fit)
  XG.4.est = predict(xgboost.fit.4, newdata = DNN.val.fit)
  
  ## RF
  RF.1.est = predict(rf.fit.1, newdata = data.svm.val)
  RF.2.est = predict(rf.fit.2, newdata = data.svm.val)
  RF.3.est = predict(rf.fit.3, newdata = data.svm.val)
  RF.4.est = predict(rf.fit.4, newdata = data.svm.val)
  
  val.para.grid$DNN_bias[val.ind] = mean((DNN.T1.T2.est - rate.delta.val))
  val.para.grid$DNN[val.ind] = mean((DNN.T1.T2.est - rate.delta.val)^2)
  
  val.para.grid$SVM_1[val.ind] = mean((SVM.1.est - rate.delta.val)^2) 
  val.para.grid$SVM_2[val.ind] = mean((SVM.2.est - rate.delta.val)^2) 
  val.para.grid$SVM_3[val.ind] = mean((SVM.3.est - rate.delta.val)^2) 
  val.para.grid$SVM_4[val.ind] = mean((SVM.4.est - rate.delta.val)^2) 
  
  val.para.grid$XG_1[val.ind] = mean((XG.1.est - rate.delta.val)^2) 
  val.para.grid$XG_2[val.ind] = mean((XG.2.est - rate.delta.val)^2) 
  val.para.grid$XG_3[val.ind] = mean((XG.3.est - rate.delta.val)^2) 
  val.para.grid$XG_4[val.ind] = mean((XG.4.est - rate.delta.val)^2) 
  
  val.para.grid$RF_1[val.ind] = mean((RF.1.est - rate.delta.val)^2) 
  val.para.grid$RF_2[val.ind] = mean((RF.2.est - rate.delta.val)^2) 
  val.para.grid$RF_3[val.ind] = mean((RF.3.est - rate.delta.val)^2) 
  val.para.grid$RF_4[val.ind] = mean((RF.4.est - rate.delta.val)^2) 
  
  # val.para.grid$comb_1[val.ind] = mean((naive.1.est - rate.delta.val)^2)
  # val.para.grid$comb_2[val.ind] = mean((naive.2.est - rate.delta.val)^2)
  # val.para.grid$comb_3[val.ind] = mean((naive.3.est - rate.delta.val)^2)
}
print(val.para.grid)

########################################################################
library(xtable)

latex.out.1 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$rate_pbo),
  "k" = sprintf("%.2f", val.para.grid$rate_delta),
  "DNN" = paste0(sprintf("%.4f", val.para.grid$DNN)),
  "SVM_1" = paste0(sprintf("%.4f", val.para.grid$SVM_1)),
  "SVM_2" = paste0(sprintf("%.4f", val.para.grid$SVM_2)),
  "SVM_3" = paste0(sprintf("%.4f", val.para.grid$SVM_3)),
  "SVM_4" = paste0(sprintf("%.4f", val.para.grid$SVM_4))
)

latex.out.2 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$rate_pbo),
  "k" = sprintf("%.2f", val.para.grid$rate_delta),
  "DNN" = paste0(sprintf("%.4f", val.para.grid$DNN)),
  "XG_1" = paste0(sprintf("%.4f", val.para.grid$XG_1)),
  "XG_2" = paste0(sprintf("%.4f", val.para.grid$XG_2)),
  "XG_3" = paste0(sprintf("%.4f", val.para.grid$XG_3)),
  "XG_4" = paste0(sprintf("%.4f", val.para.grid$XG_4))
)

latex.out.3 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$rate_pbo),
  "k" = sprintf("%.2f", val.para.grid$rate_delta),
  "DNN" = paste0(sprintf("%.4f", val.para.grid$DNN)),
  "RF_1" = paste0(sprintf("%.4f", val.para.grid$RF_1)),
  "RF_2" = paste0(sprintf("%.4f", val.para.grid$RF_2)),
  "RF_3" = paste0(sprintf("%.4f", val.para.grid$RF_3)),
  "RF_4" = paste0(sprintf("%.4f", val.para.grid$RF_4))
)

print(xtable(latex.out.1), include.rownames = FALSE)
print(xtable(latex.out.2), include.rownames = FALSE)
print(xtable(latex.out.3), include.rownames = FALSE)





























