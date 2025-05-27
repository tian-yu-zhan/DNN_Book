
library(keras)
library(reticulate)
library(tensorflow)
library(keras)
library(tibble)
library(car)
library(kernlab)      
library(e1071) 
library(xgboost)
library(randomForest)

###############################################################################
set.seed(1) 
alpha = 0.05 
# alpha.working = 0.048
n.ind = 500
n.train.H0.itt = 1*10^4
n.train.H1.itt = 1*10^4  
n.test.H0.inner.itt = 10^6

n.1 = 20

######################################################
theta.train.vec = runif(n.ind, min = 0.5, max = 10)
k.train.vec = runif(n.ind, min = 0, max = 1)

n.train.itt = n.train.H0.itt + n.train.H1.itt
data.train = matrix(NA, nrow = n.ind*n.train.itt, ncol = 6)
data.train.label = rep(NA, n.ind*n.train.itt)

get.data.case.func = function(theta.grp.1.in, 
                              theta.grp.2.in, 
                              k.in,
                              n.in,
                              if.test){
  
  ## simulate data
  data.grp.1.in = runif(n.in, min = (1-k.in)*theta.grp.1.in, 
                        max =  (1+k.in)*theta.grp.1.in)
  # data.grp.1.summary = c(summary(data.grp.1.in), sd(data.grp.1.in))
  data.grp.1.summary = c(min(data.grp.1.in), max(data.grp.1.in))
  
  data.grp.2.in = runif(n.in, min = (1-k.in)*theta.grp.2.in, 
                        max =  (1+k.in)*theta.grp.2.in)
  # data.grp.2.summary = c(summary(data.grp.2.in), sd(data.grp.2.in))
  data.grp.2.summary = c(min(data.grp.2.in), max(data.grp.2.in))
  
  data.grp.12.in = c(data.grp.1.in, data.grp.2.in)
  
  data.return.vec = c(data.grp.1.summary, 
                      data.grp.2.summary, 
                      k.in,
                      mean(data.grp.12.in)
                      )
  
  ## if add t test and wilcox test
  if (if.test){
    t.test.p.value = t.test(x = data.grp.2.in, y = data.grp.1.in, 
                            alternative = "greater")$p.value
    
  } else{
    t.test.p.value = NULL
  }
  
  new.list = list("data" = data.return.vec,
                  "test" = t.test.p.value)
  return(new.list)
}

###############################################################################
## generate training data for the first DNN
for (ind in 1:n.ind){
  print(paste("train ind:", ind))

  theta.grp.1.train = theta.train.vec[ind] 
  k.train = k.train.vec[ind] 
  
  sd.temp = (2*k.train*theta.grp.1.train)/sqrt(12) 
  
  delta.train = qnorm(alpha, sd = sqrt(2*sd.temp^2/n.1), lower.tail = FALSE)-
    qnorm(0.45, sd = sqrt(2*sd.temp^2/n.1),lower.tail = FALSE)
  
  theta.grp.2.train = theta.train.vec[ind] + delta.train
    
  data.train.H0 = t(sapply(1:n.train.H0.itt, 
               function(x){get.data.case.func(
                 theta.grp.1.in = theta.grp.1.train, 
                 theta.grp.2.in = theta.grp.1.train, 
                 k.in = k.train,
                 n.in = n.1,
                 if.test = FALSE)$data}))

  data.train.H1 = t(sapply(1:n.train.H1.itt, 
                           function(x){get.data.case.func(
                             theta.grp.1.in = theta.grp.1.train, 
                             theta.grp.2.in = theta.grp.2.train, 
                             k.in = k.train,
                             n.in = n.1,
                             if.test = FALSE)$data}))
  
  ## aggregate training data
data.train.pre = data.frame(rbind(data.train.H0, data.train.H1))
  ## labels for the training data
data.train.label.pre = c(rep(0, n.train.H0.itt), rep(1, n.train.H1.itt))

data.train[(1:n.train.itt)+(ind-1)*n.train.itt, ] = as.matrix(data.train.pre)
data.train.label[(1:n.train.itt)+(ind-1)*n.train.itt] = data.train.label.pre

}


# test = xgboost(data = as.matrix(data.train[, 1:5]), 
#                         label = data.train.label, 
#                         max.depth = 6,
#                         eta = 0.3,
#                         gamma = 0, 
#                         nrounds = 300,
#                         objective = "reg:squarederror")

###########################################################################
data.train =  as_tibble(data.train[, 1:5])
data.train.scale =scale(data.train)

col_means_train <- attr(data.train.scale, "scaled:center")
col_stddevs_train <- attr(data.train.scale, "scaled:scale")

set_random_seed(1)
model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

dnn_history = model %>% fit(
  data.train.scale,
  data.train.label,
  epochs = 20,
  batch_size = 10^4,
  validation_split = 0
)

print(dnn_history)

############################################################################
cutoff.out.vec = rep(NA, n.ind)
  
for (cutoff.ind in 1:n.ind){
  
  print(cutoff.ind)
  theta.grp.1.cutoff = theta.train.vec[cutoff.ind] 
  k.cutoff = k.train.vec[cutoff.ind] 
  
  data.cutoff.H0 = t(sapply(1:n.test.H0.inner.itt, 
                            function(x){get.data.case.func(
                              theta.grp.1.in = theta.grp.1.cutoff, 
                              theta.grp.2.in = theta.grp.1.cutoff, 
                              k.in = k.cutoff,
                              n.in = n.1,
                              if.test = FALSE)$data}))
  
  data.cutoff.H0.scale = scale(data.cutoff.H0[, 1:5],
                               center = col_means_train, 
                               scale = col_stddevs_train)
  
  null.rate.cutoff.pred = model %>% predict(data.cutoff.H0.scale)
  null.cutoff.pred = log(null.rate.cutoff.pred/(1-null.rate.cutoff.pred))
  
  cutoff.out.vec[cutoff.ind] = 
    (as.numeric(quantile(null.cutoff.pred, prob = 1-alpha, type=3)))
 
}

data.cutoff.train = data.frame(
  "theta" = theta.train.vec,
  "k" = k.train.vec)
data.cutoff.train =  as_tibble(data.cutoff.train)
data.cutoff.train.scale =scale(data.cutoff.train)

col_means_cutoff_train <- attr(data.cutoff.train.scale, "scaled:center")
col_stddevs_cutoff_train <- attr(data.cutoff.train.scale, "scaled:scale")

#### DNN
set_random_seed(1)
model.cutoff <- keras_model_sequential()

model.cutoff %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 100, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = 'linear')

model.cutoff %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = list('mse')
)

dnn_cutoff_history = model.cutoff %>% fit(
  data.cutoff.train.scale,
  cutoff.out.vec,
  epochs = 10^3,
  batch_size = 10,
  validation_split = 0
)

print(dnn_cutoff_history)

#### SVM
data.svm = data.frame("theta" = theta.train.vec,
                      "k" = k.train.vec, 
                      "y" = cutoff.out.vec)
svm.fit.1 =  svm(y ~ theta + k, data =data.svm, scale = TRUE,
                 kernel = "linear")
svm.fit.2 =  svm(y ~ theta + k, data =data.svm, scale = TRUE,
                 kernel = "polynomial")
svm.fit.3 =  svm(y ~ theta + k, data =data.svm, scale = TRUE,
                 kernel = "radial")
svm.fit.4 =  svm(y ~ theta + k, data =data.svm, scale = TRUE,
                 kernel = "sigmoid")

### XGBoost
xgboost.fit.1 = xgboost(data = as.matrix(data.svm[,c("theta", "k")]), 
                        label = as.numeric(data.svm$y), 
                     max.depth = 6, 
                     eta = 0.3,
                     gamma = 0, 
                     nrounds = 1000,
                     objective = "reg:squarederror")

xgboost.fit.2 = xgboost(data = as.matrix(data.svm[,c("theta", "k")]), 
                        label = as.numeric(data.svm$y), 
                        max.depth = 6, 
                        eta = 0.3,
                        gamma = 1, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

xgboost.fit.3 = xgboost(data = as.matrix(data.svm[,c("theta", "k")]), 
                        label = as.numeric(data.svm$y), 
                        max.depth = 6, 
                        eta = 1,
                        gamma = 0, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

xgboost.fit.4 = xgboost(data = as.matrix(data.svm[,c("theta", "k")]), 
                        label = as.numeric(data.svm$y), 
                        max.depth = 6, 
                        eta = 1,
                        gamma = 1, 
                        nrounds = 1000,
                        objective = "reg:squarederror")

## random forest
rf.fit.1 = randomForest(y ~ theta + k, data =data.svm, 
                        ntree = 500, nodesize = 5)
rf.fit.2 = randomForest(y ~ theta + k, data =data.svm, 
                        ntree = 500, nodesize = 10)
rf.fit.3 = randomForest(y ~ theta + k, data =data.svm, 
                        ntree = 1000, nodesize = 5)
rf.fit.4 = randomForest(y ~ theta + k, data =data.svm, 
                        ntree = 1000, nodesize = 10)

###########################################################################
theta.diff.val.vec = c(rep(2, 2), rep(7, 2))
k.val.vec = rep(c(rep(0.3, 1), rep(0.7, 1)),2)
delta.prop.val.vec = rep(0, 4)
n.val.itt = 10^5

val.para.grid = data.frame("theta" = theta.diff.val.vec,
                           "k" = k.val.vec,
                           "prop" = delta.prop.val.vec)

  val.para.grid$t_test = 
    val.para.grid$RF_4 = val.para.grid$RF_3 =
    val.para.grid$RF_2 = val.para.grid$RF_1 =
    val.para.grid$XG_4 = val.para.grid$XG_3 = 
    val.para.grid$XG_2 = val.para.grid$XG_1 = 
    val.para.grid$SVM_4 = 
    val.para.grid$SVM_3 = val.para.grid$SVM_2 = 
    val.para.grid$SVM_1 = 
    val.para.grid$DNN_power = 
  val.para.grid$trt_diff = NA
n.val.ind = dim(val.para.grid)[1]

## evaluate each scenario
for (val.ind in c(1:4)){
  set.seed(val.ind)
  print(val.ind)
  
  theta.val = val.para.grid$theta[val.ind]
  k.val = val.para.grid$k[val.ind]
  sd.val = (2*k.val*theta.val)/sqrt(12) 
  
  mean.diff.val = val.para.grid$prop[val.ind]*
    (qnorm(alpha, sd = sqrt(2*sd.val^2/n.1), lower.tail = FALSE)-
    qnorm(0.45, sd = sqrt(2*sd.val^2/n.1),lower.tail = FALSE))
  
  val.para.grid$trt_diff[val.ind] = mean.diff.val
  mean.grp.1.val = theta.val
  mean.grp.2.val = theta.val + mean.diff.val
  
  validation.mat = t(sapply(1:n.val.itt, function(x){

    data.val.fit = get.data.case.func(
      theta.grp.1.in = mean.grp.1.val, 
      theta.grp.2.in = mean.grp.2.val, 
      k.in = k.val,
      n.in = n.1,
      if.test = TRUE)
    
    ## return the data and p-values from other methods
    val.return.vec = c(data.val.fit$data,
                       data.val.fit$test)
    return(val.return.vec)
  }))
  
  validation.data.input = as.matrix(validation.mat[, c(1:5)])

  validation.test.output = validation.mat[, 7]
  
  validation.data.input.scale = scale(validation.data.input,
                                      center = col_means_train, 
                                      scale = col_stddevs_train)
  
  data.rate.val = model %>% predict((validation.data.input.scale))
  data.stats.val = as.numeric(log(data.rate.val/(1-data.rate.val)))
  
  validation.data.cutoff.scale = scale(validation.mat[, c(6, 5)],
                                       center = col_means_cutoff_train, 
                                       scale = col_stddevs_cutoff_train)
  data.cutoff.val = model.cutoff %>% predict((validation.data.cutoff.scale))
  
 val.para.grid[val.ind, c("DNN_power")] = mean(data.stats.val>=data.cutoff.val)
  
 ## svm
 data.svm.val = data.frame("theta" = validation.mat[, 6],
                           "k" = validation.mat[, 5])
 
 svm.cutof.val.1 = predict(svm.fit.1, newdata = data.svm.val)
 val.para.grid[val.ind, c("SVM_1")] = mean(data.stats.val>=(svm.cutof.val.1+0))
 
 svm.cutof.val.2 = predict(svm.fit.2, newdata = data.svm.val)
 val.para.grid[val.ind, c("SVM_2")] = mean(data.stats.val>=(svm.cutof.val.2+0))
 
 svm.cutof.val.3 = predict(svm.fit.3, newdata = data.svm.val)
 val.para.grid[val.ind, c("SVM_3")] = mean(data.stats.val>=(svm.cutof.val.3+0))
 
 svm.cutof.val.4 = predict(svm.fit.4, newdata = data.svm.val)
 val.para.grid[val.ind, c("SVM_4")] = mean(data.stats.val>=(svm.cutof.val.4+0))
 
 ## xgboost
 xgboost.cutoff.val.1 =  predict(xgboost.fit.1, 
                                 as.matrix(data.svm.val[,c("theta", "k")]))
 val.para.grid[val.ind, c("XG_1")] = mean(data.stats.val>=(xgboost.cutoff.val.1+0))
 
 xgboost.cutoff.val.2 =  predict(xgboost.fit.2, 
                                 as.matrix(data.svm.val[,c("theta", "k")]))
 val.para.grid[val.ind, c("XG_2")] = mean(data.stats.val>=(xgboost.cutoff.val.2+0))
 
 xgboost.cutoff.val.3 =  predict(xgboost.fit.3, 
                                 as.matrix(data.svm.val[,c("theta", "k")]))
 val.para.grid[val.ind, c("XG_3")] = mean(data.stats.val>=(xgboost.cutoff.val.3+0))
 
 xgboost.cutoff.val.4 =  predict(xgboost.fit.4, 
                                 as.matrix(data.svm.val[,c("theta", "k")]))
 val.para.grid[val.ind, c("XG_4")] = mean(data.stats.val>=(xgboost.cutoff.val.4+0))
 
 ## random forest
 rf.cutoff.val.1 = predict(rf.fit.1, newdata = data.svm.val)
 val.para.grid[val.ind, c("RF_1")] = mean(data.stats.val>=(rf.cutoff.val.1+0))

 rf.cutoff.val.2 = predict(rf.fit.2, newdata = data.svm.val)
 val.para.grid[val.ind, c("RF_2")] = mean(data.stats.val>=(rf.cutoff.val.2+0))
 
 rf.cutoff.val.3 = predict(rf.fit.3, newdata = data.svm.val)
 val.para.grid[val.ind, c("RF_3")] = mean(data.stats.val>=(rf.cutoff.val.3+0))
 
 rf.cutoff.val.4 = predict(rf.fit.4, newdata = data.svm.val)
 val.para.grid[val.ind, c("RF_4")] = mean(data.stats.val>=(rf.cutoff.val.4+0))
 
 
 
 
 ## t-test
  val.para.grid[val.ind, c("t_test")] = mean(validation.test.output<=alpha)
 #   apply(validation.test.output, 2, function(x){mean(x<=alpha)})


}
print(val.para.grid)

########################################################################
library(xtable)

latex.out.1 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$theta),
  "k" = sprintf("%.2f", val.para.grid$k),
  "DNN" = paste0(sprintf("%.1f", val.para.grid$DNN_power*100), "%"),
  "SVM_1" = paste0(sprintf("%.1f", val.para.grid$SVM_1*100), "%"),
  "SVM_2" = paste0(sprintf("%.1f", val.para.grid$SVM_2*100), "%"),
  "SVM_3" = paste0(sprintf("%.1f", val.para.grid$SVM_3*100), "%"),
  "SVM_4" = paste0(sprintf("%.1f", val.para.grid$SVM_4*100), "%")
)

latex.out.2 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$theta),
  "k" = sprintf("%.2f", val.para.grid$k),
  "DNN" = paste0(sprintf("%.1f", val.para.grid$DNN_power*100), "%"),
  "XG_1" = paste0(sprintf("%.1f", val.para.grid$XG_1*100), "%"),
  "XG_2" = paste0(sprintf("%.1f", val.para.grid$XG_2*100), "%"),
  "XG_3" = paste0(sprintf("%.1f", val.para.grid$XG_3*100), "%"),
  "XG_4" = paste0(sprintf("%.1f", val.para.grid$XG_4*100), "%")
)

latex.out.3 = data.frame(
  "theta" = sprintf("%.1f", val.para.grid$theta),
  "k" = sprintf("%.2f", val.para.grid$k),
  "DNN" = paste0(sprintf("%.1f", val.para.grid$DNN_power*100), "%"),
  "RF_1" = paste0(sprintf("%.1f", val.para.grid$RF_1*100), "%"),
  "RF_2" = paste0(sprintf("%.1f", val.para.grid$RF_2*100), "%"),
  "RF_3" = paste0(sprintf("%.1f", val.para.grid$RF_3*100), "%"),
  "RF_4" = paste0(sprintf("%.1f", val.para.grid$RF_4*100), "%")
)

print(xtable(latex.out.1), include.rownames = FALSE)
print(xtable(latex.out.2), include.rownames = FALSE)
print(xtable(latex.out.3), include.rownames = FALSE)
