
library(R2jags)
library(mcmcplots)
library(keras)
library(reticulate)
library(tensorflow)
library(tibble)
library(doParallel)
library(bindata)

set.seed(1)
n.vec.hist = c(100, 100, 150, 150, 200, 200) 
n.new = 150
n.hist = length(n.vec.hist) 
n.endpoint = 2
n.cluster= 4 
prior.a.trt = prior.b.trt = 1 
n.train = 2000 

fix.resp.hist.mat = matrix(c(42, 35, 59, 57, 71, 80,
                             64, 58, 92, 100, 114, 118), 
                           nrow = n.endpoint, ncol = n.hist, 
                           byrow = TRUE)
fix.resp.hist.vec = as.vector(t(fix.resp.hist.mat))

data.train = matrix(NA, nrow = n.train, 
                    ncol = 2*n.endpoint)

for (ind.train in 1:n.train){
  train.rate.new.vec.temp = c(runif(1, 0.2, 0.6),
                              runif(1, 0.4, 0.8))
  train.rate.delta.vec.temp = c(runif(1, 0, 0.1),
                                runif(1, 0, 0.1))
  
  train.rate.trt.vec.temp = pmin(0.9999, pmax(0.0001, 
                                              train.rate.new.vec.temp +
                                                train.rate.delta.vec.temp))
  
  train.resp.new.vec.temp =
    sapply(1:n.endpoint, function(x){
      rbinom(1, n.new, train.rate.new.vec.temp[x])})
  
  train.resp.trt.vec.temp = 
    sapply(1:n.endpoint, function(x){
      rbinom(1, n.new, train.rate.trt.vec.temp[x])})
  
  data.train[ind.train, ] =
    c(train.resp.new.vec.temp,
      train.resp.trt.vec.temp)
}

cl = makeCluster(n.cluster)
registerDoParallel(cl)
label.first.train = 
  foreach(ind.train=1:n.train) %dopar% { 
    
    library(R2jags)
    library(mcmcplots)
    library(keras)
    library(reticulate)
    library(tensorflow)
    library(tibble)
    library(doParallel)
    library(bindata)
    
    set.seed(ind.train)
    
    train.resp.new.vec =
      as.numeric(data.train[ind.train, 1:n.endpoint])
    train.resp.trt.vec =
      as.numeric(data.train[ind.train, (1:n.endpoint)+n.endpoint])
    
    sim.dat.jags = list(
      "n.hist" = n.hist,
      "n.endpoint" = n.endpoint,
      "resp.hist.mat" = fix.resp.hist.mat,
      "n.vec.hist" = n.vec.hist,
      "n.new" = n.new,
      "resp.new" = train.resp.new.vec,
      "inv_tau2_init" = diag(c(1,1), n.endpoint)
    )
    
    bayes.mod.params <- c("p.pred", "tau2", "mu")
    
    bayes.mod.inits <- function(){
      list("mu" = rep(0, n.endpoint))
    }
    
    bayes.mod = function() {
      for (j in 1:n.hist){
        for (i in 1:n.endpoint) {
          resp.hist.mat[i, j] ~ dbin(p[i, j], 
                                     n.vec.hist[j])
          logit(p[i, j])  =  logit_p[i, j]
        }
        logit_p[1:n.endpoint, j]   ~  dmnorm(mu[], inv_tau2[,])
      }
      
      logit_p.pred ~ 
        dmnorm(mu[], inv_tau2[,])
      
      inv_tau2[1:n.endpoint, 1:n.endpoint] ~
        dwish(inv_tau2_init[,], 3)
      
      tau2[1:n.endpoint, 1:n.endpoint] =
        inverse(inv_tau2[,])
      
      for (i in 1:n.endpoint) {
        mu[i]  ~ dnorm(0, 0.01)
      }
      
      for (i in 1:n.endpoint){
        p.pred[i] = 1 / (1 + exp(-logit_p.pred[i]))
        resp.new[i] ~ dbin(p.pred[i], n.new)
      }
    }
    
    bayes.mod.fit = jags(data = sim.dat.jags,
                         inits = bayes.mod.inits, 
                         parameters.to.save = bayes.mod.params, 
                         n.thin = 1, 
                         n.chains = 3, 
                         n.iter = 2*10^4, 
                         n.burnin = 10^4, 
                         model.file = bayes.mod, 
                         progress.bar = "none") 
    
    bayes.mod.fit = autojags(bayes.mod.fit,
                             Rhat = 1.01,
                             n.thin = 1,
                             n.update = 10,
                             n.iter = 10^4,
                             progress.bar = "none")
    
    sim.samples =
      data.frame(bayes.mod.fit$BUGSoutput$sims.matrix)
    
    p.pred.samples = matrix(NA, 
                            nrow = dim(sim.samples)[1], 
                            ncol = n.endpoint)
    
    for (n.endpoint.ind in 1:n.endpoint){
      eval(parse(text=paste0(
        "p.pred.samples[,n.endpoint.ind]=sim.samples$p.pred.",n.endpoint.ind, ".")))
    }
    
    p.trt.pred.samples = matrix(NA, 
                                nrow = dim(sim.samples)[1], 
                                ncol = n.endpoint)
    
    for (n.endpoint.ind in 1:n.endpoint){
      p.trt.pred.samples[, n.endpoint.ind] =
        rbeta(dim(sim.samples)[1],
              shape1 = train.resp.trt.vec[n.endpoint.ind]+
                prior.a.trt,
              shape2 = n.new -
                train.resp.trt.vec[n.endpoint.ind]+prior.b.trt)
    }
    
    label.train.return = c(sapply(1:n.endpoint,
                                  function(x){mean(p.trt.pred.samples[, x]>
                                                     p.pred.samples[, x])}),
                           apply(p.pred.samples, 2, mean))
    
    return(label.train.return)
  }
stopCluster(cl)

label.train.all = matrix(unlist(label.first.train), 
                         nrow = n.train, ncol = 2*n.endpoint, byrow = TRUE)

data.train = as_tibble(data.train)
data.train.scale =scale(data.train[1:(n.train*0.8), ])
label.train = label.train.all[1:(n.train*0.8), ]

col_means_train = 
  attr(data.train.scale, "scaled:center")
col_stddevs_train = 
  attr(data.train.scale, "scaled:scale")

set_random_seed(1)
model <- keras_model_sequential()

model %>%
  layer_dense(units = 60, activation = "sigmoid") %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 60, activation = "sigmoid") %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 2*n.endpoint, 
              activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = list('mse'))

dnn_history = model %>% fit(
  data.train.scale,
  label.train,
  epochs = 100,
  batch_size = 100,
  validation_split = 0
)

train.pred = model %>% predict(data.train.scale)

val.input =
  scale(data.train[(n.train*0.8+1):(n.train), ], 
        center = col_means_train, 
        scale = col_stddevs_train)

val.pred = model %>% predict(val.input)

output.mat = matrix(NA, nrow = 2, ncol = 4)

output.mat[1, ] = sapply(1:(2*n.endpoint),
                         function(x){mean((train.pred[,x]-
                                             label.train[,x])^2)})

output.mat[2, ] = sapply(1:(2*n.endpoint),
                         function(x){mean((val.pred[,x]-
                                             label.train.all[(n.train*0.8+1):(n.train), x])^2)})




