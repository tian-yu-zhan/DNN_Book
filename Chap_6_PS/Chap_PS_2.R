
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)

n.train = 1000 
n.boot = 10^5
n = 50
alpha = 0.05
input.mat = matrix(NA, nrow = n.train, ncol = 7)
label.vec = rep(NA, n.train)

time.count.1 = Sys.time()

for (ind.train in 1:n.train){
  set.seed(ind.train)
  data.input = runif(n, -10, 10)
  
  input.mat[ind.train, ] = c(mean(data.input),
                             median(data.input),
                             min(data.input),
                             max(data.input),
                             quantile(data.input, 0.25),
                             quantile(data.input, 0.75),
                             sd(data.input))
  
  data.boot = sapply(1:n.boot, function(ind.boot){
    data.boot.temp = sample(data.input, 
                            n, replace=TRUE)
    mean.temp = mean(data.boot.temp)
    sigmoid.temp = 1/(1+exp(-mean.temp))
    return(sigmoid.temp)
  })
  
  label.vec[ind.train] = quantile(data.boot, alpha)
}
time.train.diff = difftime(Sys.time(), time.count.1, units="secs")

data.train =  as_tibble(input.mat)
data.train.scale =scale(data.train[1:(n.train*0.8), ])
label.train = label.vec[1:(n.train*0.8)]

col_means_train = 
  attr(data.train.scale, "scaled:center")
col_stddevs_train = 
  attr(data.train.scale, "scaled:scale")

set_random_seed(1)
model = keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = 'linear')

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = list('mse')
)

dnn_history = model %>% fit(
  data.train.scale,
  label.train,
  epochs = 100,
  batch_size = 100,
  validation_split = 0
)

train.pred = model %>% predict(data.train.scale)

time.DNN.diff = difftime(Sys.time(), time.count.1, units="secs")

time.count.2 = Sys.time()
val.input = scale(
  data.train[(n.train*0.8+1):(n.train), ], 
  center = col_means_train, 
  scale = col_stddevs_train)

val.pred = model %>% predict(val.input)
time.val.diff = difftime(Sys.time(), time.count.2, units="secs")

label.val = label.vec[(n.train*0.8+1):(n.train)]

print(mean((train.pred - label.train)^2))
print(mean((val.pred - label.val)^2))

























