
library(keras)
library(reticulate)
library(tensorflow)
library(keras)
library(tibble)
library(car)

###############################################################################
set.seed(1) 
alpha = 0.05 
n.ind = 10^3
n.train.H0.itt = 1*10^4
n.train.H1.itt = 1*10^4  
n.test.H0.inner.itt = 10^5

n.1 = 200

######################################################
mean.grp.1.train.vec = runif(n.ind, min = -1, max = 1)
sd.train.vec = runif(n.ind, min = 0.5, max = 2.5)

n.train.itt = n.train.H0.itt + n.train.H1.itt
data.train = matrix(NA, nrow = n.ind*n.train.itt, ncol = 6)
data.train.label = rep(NA, n.ind*n.train.itt)

get.data.case.func = function(mean.grp.1.train.in, 
                              mean.grp.2.train.in, 
                              sd.train.in,
                              prop.unif.in, 
                              if.test){
  
  ## simulate first stage data
  # data.grp.1 = (1-prop.unif.in)*rnorm(n.1, mean.grp.1.train.in, sd.train.in)+
  #   prop.unif.in*runif(n.1, min = mean.grp.1.train.in - sqrt(sd.train.in^2*12)/2, 
  #                      max = mean.grp.1.train.in + sqrt(sd.train.in^2*12)/2)
  # data.grp.2 = (1-prop.unif.in)*rnorm(n.1, mean.grp.2.train.in, sd.train.in)+
  #   prop.unif.in*runif(n.1, min = mean.grp.2.train.in - sqrt(sd.train.in^2*12)/2, 
  #                      max = mean.grp.2.train.in + sqrt(sd.train.in^2*12)/2)
  
  data.grp.1 = c(rnorm((1-prop.unif.in)*n.1, mean.grp.1.train.in, sd.train.in),
    runif(prop.unif.in*n.1, min = mean.grp.1.train.in - sqrt(sd.train.in^2*12)/2,
                       max = mean.grp.1.train.in + sqrt(sd.train.in^2*12)/2)
  )
  data.grp.2 = c(rnorm((1-prop.unif.in)*n.1, mean.grp.2.train.in, sd.train.in),
                 runif(prop.unif.in*n.1, min = mean.grp.2.train.in - sqrt(sd.train.in^2*12)/2,
                       max = mean.grp.2.train.in + sqrt(sd.train.in^2*12)/2)
  )
  
  data.return.vec = c(mean(data.grp.1),
                      mean(data.grp.2),
                      sd(data.grp.1),
                      sd(data.grp.2),
                      mean(c(data.grp.1, data.grp.2)),
                      sd(c(data.grp.1, data.grp.2))
  )
  
  if (if.test){
    p.1.value = t.test(x = data.grp.1, y = data.grp.2,
                          alternative = "less",
                          var.equal = TRUE)$p.value
    
  } else{
    p.1.value = NULL
  }
  
  new.list = list("data" = data.return.vec,
                  "test" = c(p.1.value))
  return(new.list)
}

###############################################################################
## generate training data for the first DNN
for (ind in 1:n.ind){
  print(paste("train ind:", ind))

  mean.grp.1.train = mean.grp.1.train.vec[ind]
  # mean.grp.2.train = mean.grp.1.train + mean.diff.train.vec[ind] 
  
  sd.train = sd.train.vec[ind]
  
  mean.grp.2.train = mean.grp.1.train + (qnorm(alpha, 
        sd = sqrt(2*sd.train^2/n.1), 
        lower.tail = FALSE)-
 qnorm(0.9, sd = sqrt(2*sd.train^2/n.1),
       lower.tail = FALSE))
    
  data.train.H0 = t(sapply(1:n.train.H0.itt, 
               function(x){get.data.case.func(
                 mean.grp.1.train.in = mean.grp.1.train, 
                 mean.grp.2.train.in = mean.grp.1.train, 
                 sd.train.in = sd.train,
                 prop.unif.in = 0, 
                 if.test = FALSE)$data}))

  data.train.H1 = t(sapply(1:n.train.H1.itt, 
                           function(x){get.data.case.func(
                             mean.grp.1.train.in = mean.grp.1.train, 
                             mean.grp.2.train.in = mean.grp.2.train, 
                             sd.train.in = sd.train,
                             prop.unif.in = 0, 
                             if.test = FALSE)$data}))
  
  ## aggregate training data
data.train.pre = data.frame(rbind(data.train.H0, data.train.H1))
  ## labels for the training data
data.train.label.pre = c(rep(0, n.train.H0.itt), rep(1, n.train.H1.itt))

data.train[(1:n.train.itt)+(ind-1)*n.train.itt, ] = as.matrix(data.train.pre)
data.train.label[(1:n.train.itt)+(ind-1)*n.train.itt] = data.train.label.pre

}

###########################################################################
data.train =  as_tibble(data.train[, 1:4])
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
  mean.cutoff = mean.grp.1.train.vec[ind]
  sd.cutoff = sd.train.vec[cutoff.ind] ## \theta_{12} under null
  
  data.cutoff.H0 = t(sapply(1:n.test.H0.inner.itt, 
                            function(x){get.data.case.func(
                              mean.grp.1.train.in = mean.cutoff, 
                              mean.grp.2.train.in = mean.cutoff, 
                              sd.train.in = sd.cutoff,
                              prop.unif.in = 0, 
                              if.test = FALSE)$data}))
  
  data.cutoff.H0.scale = scale(data.cutoff.H0[, 1:4],
                               center = col_means_train, 
                               scale = col_stddevs_train)
  
  null.rate.cutoff.pred = model %>% predict(data.cutoff.H0.scale)
  null.cutoff.pred = log(null.rate.cutoff.pred/(1-null.rate.cutoff.pred))
  
  cutoff.out.vec[cutoff.ind] = 
    (as.numeric(quantile(null.cutoff.pred, prob = 1-alpha, type=3)))
 
}

data.cutoff.train = data.frame("sd" = sd.train.vec)
data.cutoff.train =  as_tibble(data.cutoff.train)
data.cutoff.train.scale =scale(data.cutoff.train)

col_means_cutoff_train <- attr(data.cutoff.train.scale, "scaled:center")
col_stddevs_cutoff_train <- attr(data.cutoff.train.scale, "scaled:scale")

set_random_seed(1)
model.cutoff <- keras_model_sequential()

model.cutoff %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'linear')

model.cutoff %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = list('mse')
)

dnn_cutoff_history = model.cutoff %>% fit(
  data.cutoff.train.scale,
  cutoff.out.vec,
  epochs = 1000,
  batch_size = 100,
  validation_split = 0
)

print(dnn_cutoff_history)

###########################################################################
mean.prop.val.vec = rep(c(0, 0.9, 1, 1.1), 2)
sd.val.vec = c(rep(1.3, 4), rep(1.7, 4))

prop.unif = 0 ## normal data
# prop.unif = 0.2 ## mixture data with 0.2
# prop.unif = 0.5 ## mixture data with 0.5

# mean.prop.val.vec = rep(0, 7)
# sd.val.vec = c(0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1)
n.val.itt = 10^5

val.para.grid = data.frame("mean_diff_prop" = mean.prop.val.vec,
                           "sd" = sd.val.vec)

val.para.grid$mean_diff = 
val.para.grid$t_power = val.para.grid$DNN_power = NA
n.val.ind = dim(val.para.grid)[1]

## evaluate each scenario
for (val.ind in 1:n.val.ind){
  set.seed(val.ind)
  print(val.ind)
  
  sd.val = val.para.grid$sd[val.ind] 
  
  mean.diff.val.base = (qnorm(alpha, 
         sd = sqrt(2*sd.val^2/n.1), 
         lower.tail = FALSE)-
      qnorm(0.9, sd = sqrt(2*sd.val^2/n.1),
            lower.tail = FALSE))
  
  mean.diff.val = val.para.grid$mean_diff_prop[val.ind]*mean.diff.val.base
  val.para.grid$mean_diff[val.ind] = mean.diff.val
  
  validation.mat = t(sapply(1:n.val.itt, function(x){

    data.val.fit = get.data.case.func(
      mean.grp.1.train.in = 0, 
      mean.grp.2.train.in = mean.diff.val, 
      sd.train.in = sd.val,
      prop.unif.in = prop.unif, 
      if.test = TRUE)
    
    ## return the data and p-values from other methods
    val.return.vec = c(data.val.fit$data,
                       data.val.fit$test)
    return(val.return.vec)
  }))
  
  validation.data.input = as.matrix(validation.mat[, c(1:4)])

  validation.test.output = validation.mat[, 7]
  
  validation.data.input.scale = scale(validation.data.input,
                                      center = col_means_train, 
                                      scale = col_stddevs_train)
  
  data.rate.val = model %>% predict((validation.data.input.scale))
  data.stats.val = as.numeric(log(data.rate.val/(1-data.rate.val)))
  
  validation.data.cutoff.scale = scale(validation.mat[, 6],
                                       center = col_means_cutoff_train, 
                                       scale = col_stddevs_cutoff_train)
    
  data.cutoff.val = model.cutoff %>% predict((validation.data.cutoff.scale))
  
 val.para.grid[val.ind, c("DNN_power")] = mean(data.stats.val>=data.cutoff.val)
 
 val.para.grid[val.ind, c("t_power")] = mean(validation.test.output<=alpha)

}
print(val.para.grid)

########################################################################
library(xtable)

latex.out = data.frame(
  "sd" = sprintf("%.1f", val.para.grid$sd),
  "diff" = sprintf("%.2f", val.para.grid$mean_diff),
  "DNN" = paste0(sprintf("%.1f", val.para.grid$DNN_power*100), "%"),
  "Com_1" = paste0(sprintf("%.1f", val.para.grid$t_power*100), "%")
)

print(xtable(latex.out), include.rownames = FALSE)

