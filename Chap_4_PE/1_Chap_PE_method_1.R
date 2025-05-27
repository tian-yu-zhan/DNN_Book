
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)

set.seed(1) 
n.first.itt = 10^3 
n.itt = 10^5
n.cluster = 8
n.stage.1 = 200
n.stage.2.1 = 100 
n.stage.2.2 = 400 
ratio.stage.2 = 1/4
theta.cutoff = 0.15

DNN.first.pbo.rate.vec = 
  runif(n.first.itt, min = 0.1, max = 0.5)
DNN.first.delta.rate.vec = 
  runif(n.first.itt, min = -0.1, max = 0.3)

adaptive.bin.data.func = function(rate.pbo.in,
                                  rate.trt.in){
  rand.1.pbo = rbinom(n.stage.1/2, 1, rate.pbo.in)
  rand.1.trt = rbinom(n.stage.1/2, 1, rate.trt.in)
  rand.1.delta = mean(rand.1.trt) - mean(rand.1.pbo)
  
  adap.ind = (rand.1.delta > theta.cutoff)
  
  if (adap.ind){
    rand.2.pbo = rbinom(n.stage.2.2*ratio.stage.2, 1,
                        rate.pbo.in)
    rand.2.trt = rbinom(n.stage.2.2*(1-ratio.stage.2), 
                        1, rate.trt.in)
  } else {
    rand.2.pbo = rbinom(n.stage.2.1*(1-ratio.stage.2), 
                        1, rate.pbo.in)
    rand.2.trt = rbinom(n.stage.2.1*ratio.stage.2, 
                        1, rate.trt.in)
  }
  
  rand.2.delta = mean(rand.2.trt) - mean(rand.2.pbo)
  
  new.list = c(rand.1.delta,
               rand.2.delta,
               0.5*mean(rand.1.pbo)+0.5*mean(rand.2.pbo)
  )
  
  return(new.list)
}

cl = makeCluster(n.cluster)
registerDoParallel(cl)

DNN.first.label = foreach(first.ind=1:n.first.itt)%dopar% {
  
  library(keras)
  library(reticulate)
  library(tensorflow)
  library(tibble)
  set.seed(first.ind)
  
  rate.pbo = DNN.first.pbo.rate.vec[first.ind]
  rate.delta = DNN.first.delta.rate.vec[first.ind]
  rate.trt = rate.pbo + rate.delta
  
  DNN.first.train = t(sapply(1:n.itt, 
                             function(temp.ind){adaptive.bin.data.func(
                               rate.pbo.in = rate.pbo, 
                               rate.trt.in = rate.trt)}))
  
  t1 = 0.5*DNN.first.train[, 1] +
    0.5*DNN.first.train[, 2]
  t2 = DNN.first.train[, 1]
  
  first.w1 = mean((t2-t1)*(t2-rate.delta))/
    mean((t2-t1)^2)
  
  return(first.w1)
}
stopCluster(cl)

data.train =  as_tibble(cbind(DNN.first.pbo.rate.vec,
                              DNN.first.delta.rate.vec))
data.train.scale =scale(data.train)
data.train.label = unlist(DNN.first.label)

col_means_train = attr(data.train.scale,
                       "scaled:center")
col_stddevs_train = attr(data.train.scale,
                         "scaled:scale")

set_random_seed(1)
model = keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'linear')

model %>% compile(optimizer =
                    optimizer_rmsprop(learning_rate = 0.001),
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

rate.pbo.val.vec = c(0.3, 0.3, 0.4, 0.4)
rate.delta.val.vec = c(0, 0.2, 0, 0.2)
n.val.itt = 10^5
n.val.ind = length(rate.pbo.val.vec)

val.para.grid = data.frame("rate_pbo" =
                             rate.pbo.val.vec,
                           "rate_delta" = rate.delta.val.vec)

val.para.grid$comb_3 = val.para.grid$comb_2 = 
  val.para.grid$comb_1 = val.para.grid$DNN = 
  val.para.grid$DNN_bias = NA

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
  
  delta.1.est = DNN.val.fit[, 1]
  delta.2.est = DNN.val.fit[, 2]
  pbo.rate.est = DNN.val.fit[, 3] 
  
  DNN.T1.est = 0.5*delta.1.est + 0.5*delta.2.est
  DNN.T2.est = delta.1.est
  
  naive.1.est = 0.8*delta.1.est + 0.2*delta.2.est 
  naive.2.est = DNN.T1.est
  naive.3.est = 0.2*delta.1.est + 0.8*delta.2.est
  
  data.DNN.w1.est = cbind(pbo.rate.est, 
                          DNN.T1.est)
  
  DNN.test.data.scale = scale(data.DNN.w1.est,
                              center = col_means_train, 
                              scale = col_stddevs_train)
  
  DNN.w1.est = 
    model %>% predict(DNN.test.data.scale)
  
  DNN.T1.T2.est = 
    DNN.w1.est*DNN.T1.est + (1-DNN.w1.est)*DNN.T2.est
  
  val.para.grid$DNN_bias[val.ind] =
    mean((DNN.T1.T2.est - rate.delta.val))
  val.para.grid$DNN[val.ind] =
    mean((DNN.T1.T2.est - rate.delta.val)^2)
  val.para.grid$comb_1[val.ind] =
    mean((naive.1.est - rate.delta.val)^2)
  val.para.grid$comb_2[val.ind] =
    mean((naive.2.est - rate.delta.val)^2)
  val.para.grid$comb_3[val.ind] =
    mean((naive.3.est - rate.delta.val)^2)
}

########################################################################
library(xtable)
print(xtable(val.para.grid, digits = c(1, 1, 1, 5, 4, 4, 4, 4)),
      include.rownames=FALSE)











































