
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

theta.diff.val.vec = c(rep(2, 2), rep(7, 2))
k.val.vec = rep(c(rep(0.3, 1), rep(0.7, 1)),2)

final.output = data.frame("theta" = theta.diff.val.vec,
                           "k" = k.val.vec)

final.output$XG_4 = final.output$XG_3 =  
  final.output$XG_2 = final.output$XG_1 = NA

for (final.ind in 1:4){
  
  if (final.ind==1){eta.XG = 0.3; gamma.XG = 0}
  if (final.ind==2){eta.XG = 0.3; gamma.XG = 1}
  if (final.ind==3){eta.XG = 1; gamma.XG = 0}
  if (final.ind==4){eta.XG = 1; gamma.XG = 1}

time.start = Sys.time()
###############################################################################
set.seed(1) 
alpha = 0.05 
# alpha.working = 0.048
# time.fac = 15
n.ind = 500
n.train.H0.itt = 10^4
n.train.H1.itt = 10^4
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

###########################################################################
data.train =  as_tibble(data.train[, 1:5])

# data.train.svm.1 = data.frame(cbind(data.train[, 1:5], 
#                             (data.train.label)))
# colnames(data.train.svm.1) = c("x1", "x2", "x3", "x4", "x5", "y")

# svm.fit.1 = xgboost(data = as.matrix(data.train[, 1:5]), 
#         label = as.numeric(data.train.label), 
#         max.depth = 6, 
#         eta = 1,
#         gamma = 0, 
#         nrounds = 300,
#         objective = "reg:squarederror")

svm.fit.1 = xgboost(data = as.matrix(data.train[, 1:5]), 
                    label = as.numeric(data.train.label), 
                    max.depth = 6, 
                    eta = eta.XG,
                    gamma = gamma.XG, 
                    nrounds = 300,
                    objective = "binary:logistic", verbose = 0)

# svm.fit.1 =  svm(y ~ x1 + x2 + x3 + x4 + x5, data =data.train.svm.1, scale = TRUE,
#                  kernel = "radial",
#                  type = "eps-regression")

#pred.svm.1 = as.numeric(predict(svm.fit.1))-1
#print(mean(data.train.label==pred.svm.1))
#print(difftime(Sys.time(), time.start, units = "mins"))


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

  data.cutoff.H0.scale = (data.cutoff.H0[, 1:5])
  # colnames(data.cutoff.H0.scale) = c("x1", "x2", "x3", "x4", "x5")
  null.cutoff.pred.temp = predict(svm.fit.1, newdata = data.cutoff.H0.scale)
    
  null.cutoff.pred = pmax(-100, pmin(100,
                          log(null.cutoff.pred.temp/(1-null.cutoff.pred.temp))))
  
  # data.cutoff.H0.scale = scale(data.cutoff.H0[, 1:5],
  #                              center = col_means_train, 
  #                              scale = col_stddevs_train)
  # 
  # null.rate.cutoff.pred = model %>% predict(data.cutoff.H0.scale)
  # null.cutoff.pred = log(null.rate.cutoff.pred/(1-null.rate.cutoff.pred))
  
  cutoff.out.vec[cutoff.ind] = 
    (as.numeric(quantile(null.cutoff.pred, prob = 1-alpha, type=3)))
  
}

# data.train.svm.2 = data.frame("theta" = theta.train.vec,
#                       "k" = k.train.vec, 
#                       "y" = cutoff.out.vec)

svm.fit.2 = xgboost(data = as.matrix(cbind(theta.train.vec, k.train.vec)), 
                    label = as.numeric(cutoff.out.vec), 
                    max.depth = 6, 
                    eta = eta.XG,
                    gamma = gamma.XG, 
                    nrounds = 1000,
                    objective = "reg:squarederror", verbose = 0)

# svm.fit.2 =  svm(y ~ theta + k, data =data.train.svm.2, scale = TRUE,
#                  kernel = "radial")

print(difftime(Sys.time(), time.start, units = "mins"))


###########################################################################
theta.diff.val.vec = c(rep(2, 2), rep(7, 2))
k.val.vec = rep(c(rep(0.3, 1), rep(0.7, 1)),2)
delta.prop.val.vec = rep(0, 4)
n.val.itt = 10^5

val.para.grid = data.frame("theta" = theta.diff.val.vec,
                           "k" = k.val.vec,
                           "prop" = delta.prop.val.vec)

  val.para.grid$SVM = 
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
  
  # validation.data.input.scale = data.frame(validation.data.input)
  # colnames(validation.data.input.scale) = c("x1", "x2", "x3", "x4", "x5")
  data.stats.val.temp = predict(svm.fit.1, newdata = validation.data.input)

  data.stats.val = pmax(-100, pmin(100,
                log(data.stats.val.temp/(1-data.stats.val.temp))))
  
  # data.svm.val = data.frame("theta" = validation.mat[, 6],
  #                           "k" = validation.mat[, 5])
  
  data.svm.val = validation.mat[, c(6, 5)]
  
  svm.cutof.val.1 = predict(svm.fit.2, newdata = data.svm.val)
  val.para.grid[val.ind, c("SVM")] = mean(data.stats.val>=(svm.cutof.val.1))
  
  
}
print(val.para.grid)

if (final.ind==1) final.output$XG_1 = val.para.grid$SVM
if (final.ind==2) final.output$XG_2 = val.para.grid$SVM
if (final.ind==3) final.output$XG_3 = val.para.grid$SVM
if (final.ind==4) final.output$XG_4 = val.para.grid$SVM

}




########################################################################
library(xtable)

latex.out.1 = data.frame(
  "theta" = sprintf("%.1f", final.output$theta),
  "k" = sprintf("%.2f", final.output$k),
  "XG_1" = paste0(sprintf("%.1f", final.output$XG_1*100), "%"),
  "XG_2" = paste0(sprintf("%.1f", final.output$XG_2*100), "%"),
  "XG_3" = paste0(sprintf("%.1f", final.output$XG_3*100), "%"),
  "XG_4" = paste0(sprintf("%.1f", final.output$XG_4*100), "%")
)

print(xtable(latex.out.1), include.rownames = FALSE)




