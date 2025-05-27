
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)

set.seed(1) 
n.first.itt = 10^4
n.itt = 10^5
n.cluster = 8

DNN.first.pbo.rate.vec = runif(n.first.itt, min = 0, max = 0.1)
DNN.first.delta.rate.vec = runif(n.first.itt, min = 0.5, max = 0.7)
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

#################################
rate.pbo.val.vec = c(0.3, 0.3, 0.4, 0.4)
rate.delta.val.vec = c(0, 0.2, 0, 0.2)
n.val.itt = 10^5

val.para.grid = data.frame("rate_pbo" = rate.pbo.val.vec,
                           "rate_delta" = rate.delta.val.vec)

val.para.grid$comb_3 = val.para.grid$comb_2 = 
  val.para.grid$comb_1 = val.para.grid$DNN = 
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

  DNN.T1.T2.est = 
    model %>% predict(DNN.test.data.scale)
  
  val.para.grid$DNN_bias[val.ind] = mean((DNN.T1.T2.est - rate.delta.val))
  val.para.grid$DNN[val.ind] = mean((DNN.T1.T2.est - rate.delta.val)^2)
  val.para.grid$comb_1[val.ind] = mean((naive.1.est - rate.delta.val)^2)
  val.para.grid$comb_2[val.ind] = mean((naive.2.est - rate.delta.val)^2)
  val.para.grid$comb_3[val.ind] = mean((naive.3.est - rate.delta.val)^2)
}
print(val.para.grid)

########################################################################
library(xtable)
print(xtable(val.para.grid[,1:4], digits = c(1, 1, 1, 4, 4)),
      include.rownames=FALSE)





























