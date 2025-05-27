
library(tensorflow)
library(keras)
library(doParallel)
library(optimg)
library(xtable)

setwd("~/Dropbox/Research/AbbVie/Book/Code_Final/Chap_7_Int/")

###################################################################

f1.l2.func = function(w.in, a.in, b.in){
  return(w.in[1]*(((a.in)^2+(b.in)^2)) )
}

f2.l2.func = function(w.in, a.in, b.in){
  return((w.in[1]*a.in+w.in[2]*b.in))
}

f3.l2.func = function(w.in, a.in, b.in){
  return(w.in[1]*(a.in)+w.in[2]*(b.in)+w.in[3])
}

f1.l1.func = function(w.in, a.in, b.in, c.in){
  return(((qnorm(1-b.in)+qnorm(1-c.in))/a.in)^2+w.in[1]  )
}

f2.l1.func = function(w.in, a.in, b.in, c.in){
  return(((qnorm(1-b.in)+qnorm(1-c.in)+w.in[2])/a.in)^2*w.in[1])
}

f3.l1.func = function(w.in, a.in, b.in, c.in){
  return(((qnorm(1-b.in)+qnorm(1-c.in)+w.in[2])/a.in+w.in[3])^2*w.in[1])
}

f4.l1.func = function(w.in, a.in, b.in, c.in){
  return(((qnorm(1-b.in)*w.in[1]+qnorm(1-c.in)*w.in[2])/
            a.in+w.in[3])^2+w.in[4])
}

###################################################################
## parameters
set.seed(1)
n.itt = 10^3
n.1.itt = 10^4
n.cluster = 8
s1.adap.cutoff = 0.3

x1.vec = round(runif(n.itt, min = 10, max = 60)) ## first stage sample size
x2.vec = (runif(n.itt, min = 0.1, max = 0.6)) ## effect size
x3.vec = (runif(n.itt, min = 0.01, max = 0.15)) ## alpha level

cl = makeCluster(n.cluster)
registerDoParallel(cl)
pow.list = foreach(itt=1:n.itt) %dopar% {
  # for (itt in 1:n.itt){
  # print(itt)
  set.seed(itt)
  
  x1.temp = x1.vec[itt]
  x2.temp = x2.vec[itt]
  x3.temp = x3.vec[itt]
  
  pow.vec.temp = sapply(1:n.1.itt, function(itt.1){
    s1.pbo.data = rnorm(x1.temp, 0, 1)
    s1.trt.data = rnorm(x1.temp, x2.temp, 1)
    
    if ((mean(s1.trt.data)-mean(s1.pbo.data))>=s1.adap.cutoff){
      s2.n = round(x1.temp/2)
    } else {
      s2.n = x1.temp*2
    }
    
    s2.pbo.data = rnorm(s2.n, 0, 1)
    s2.trt.data = rnorm(s2.n, x2.temp, 1)
    
    stats.1 = as.numeric(t.test(s1.trt.data, s1.pbo.data,
                                alternative = "greater")$statistic)
    stats.2 = as.numeric(t.test(s2.trt.data, s2.pbo.data,
                                alternative = "greater")$statistic)
    stats.com = stats.1/sqrt(2)+stats.2/sqrt(2)
    
    # stats.com = as.numeric(t.test(s1.trt.data, s1.pbo.data,
    #                             alternative = "greater")$statistic)
    
    dec = as.numeric(pnorm(stats.com, lower.tail = FALSE)<=x3.temp)
    
    return(c(dec))
  })
  
  # pow.vec[itt] = mean(pow.vec.temp)
  return(1-mean(pow.vec.temp))
}
stopCluster(cl)

pow.vec = unlist(pow.list)

#####################################################################
## DNN training
data.input.all = cbind(x2.vec, x3.vec, pow.vec)
data.label.all = x1.vec/60

#############################################################
## 5-fold cross-validation
n.cross = 5
cross.tab.out = matrix(NA, nrow = n.cross, ncol = 2)
for (cross.ind in 1:n.cross){
  val.index = 1:(n.itt/n.cross) + (cross.ind -1)*n.itt/n.cross
  train.index = (1:n.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index]
  
  set_random_seed(1)
  model <- keras_model_sequential()
  
  model %>%
    layer_dense(units = 60, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 60, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 0.005),
    loss = 'mse',
    metrics = list('mse')
  )
  
  dnn_history = model %>% fit(
    data.input.train,
    data.label.train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0
  )
  
  # model prediction
  train.pred = as.numeric(model %>% predict(data.input.train))
  cross.tab.out[cross.ind, 1] = (mean((train.pred-data.label.train)^2))
  
  val.pred = as.numeric(model %>% predict(data.input.val))
  cross.tab.out[cross.ind, 2] = (mean((val.pred-data.label.val)^2))
}

DNN.train.MSE = mean(cross.tab.out[,1])
DNN.val.MSE = mean(cross.tab.out[,2])

#############################################################################
## simple models
f.func.value = function(l2.ind.in, l1.1.ind.in, l1.2.ind.in, 
                        w.in,
                        x.total.in){
  if (l2.ind.in==1){
    l2.f.ind = f1.l2.func
    l2.w.in = w.in[1:1]
  } else if (l2.ind.in==2){
    l2.f.ind = f2.l2.func
    l2.w.in = w.in[1:2]
  } else if (l2.ind.in==3){
    l2.f.ind = f3.l2.func
    l2.w.in = w.in[1:3]
  } 
  
  if (l1.1.ind.in==1){
    l1.1.f.ind = f1.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+1)]
  } else if (l1.1.ind.in==2){
    l1.1.f.ind = f2.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+2)]
  } else if (l1.1.ind.in==3){
    l1.1.f.ind = f3.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+3)]
  } else if (l1.1.ind.in==4){
    l1.1.f.ind = f4.l1.func
    l1.1.w.in = w.in[(length(l2.w.in)+1):(length(l2.w.in)+4)]
  } 
  
  if (l1.2.ind.in==1){
    l1.2.f.ind = f1.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+1)]
  } else if (l1.2.ind.in==2){
    l1.2.f.ind = f2.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+2)]
  } else if (l1.2.ind.in==3){
    l1.2.f.ind = f3.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+3)]
  } else if (l1.2.ind.in==4){
    l1.2.f.ind = f4.l1.func
    l1.2.w.in = w.in[(length(l2.w.in)+length(l1.1.w.in)+1):
                       (length(l2.w.in)+length(l1.1.w.in)+4)]
  } 
  
  f.total.temp = l2.f.ind(l2.w.in, 
                          l1.1.f.ind(l1.1.w.in, x.total.in[,1], x.total.in[,2],
                                     x.total.in[,3]),
                          l1.2.f.ind(l1.2.w.in, x.total.in[,1], x.total.in[,2],
                                     x.total.in[,3]))
  return(f.total.temp)
  
}

f.func.mse = function(l2.ind.mse.in, l1.1.ind.mse.in, l1.2.ind.mse.in, 
                      w.mse.in, 
                      x.total.mse.in, y.total.mse.in){
  
  f.total.temp.mse = f.func.value(l2.ind.mse.in, l1.1.ind.mse.in, 
                                  l1.2.ind.mse.in, 
                                  w.mse.in,
                                  x.total.mse.in)
  mse.out = mean((y.total.mse.in-f.total.temp.mse)^2)
  return(mse.out)
  
}

f.func.com = function(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                      x.train.com.in, y.train.com.in,
                      x.val.com.in, y.val.com.in){
  l2.w.leng = l2.ind.com.in 
  l1.1.w.leng = l1.1.ind.com.in
  l1.2.w.leng = l1.2.ind.com.in

  set.seed(1)
  opt.par.init = rnorm(l2.w.leng+l1.1.w.leng+l1.2.w.leng, 0, 0.3)

  opt.par = optimg(opt.par.init, f.func.mse,
                   l2.ind.mse.in = l2.ind.com.in, 
                   l1.1.ind.mse.in = l1.1.ind.com.in,
                   l1.2.ind.mse.in = l1.2.ind.com.in,
                   x.total.mse.in = x.train.com.in, 
                   y.total.mse.in = y.train.com.in,
                   method="ADAM")
  
  
  pred.train = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                            opt.par$par,x.train.com.in)
  pred.val = f.func.value(l2.ind.com.in, l1.1.ind.com.in, l1.2.ind.com.in, 
                          opt.par$par,x.val.com.in)
  
  return(c(mean((pred.train-y.train.com.in)^2),
           mean((pred.val-y.val.com.in)^2)))
  
}

n.cross = 5
test.tab.out = array(NA, dim = c(19, 3, n.cross))
for (cross.ind in 1:n.cross){
  val.index = 1:(n.itt/n.cross) + (cross.ind -1)*n.itt/n.cross
  train.index = (1:n.itt)[-val.index]
  
  data.input.train = data.input.all[train.index, ]
  data.label.train = data.label.all[train.index]
  
  data.input.val = data.input.all[val.index, ]
  data.label.val = data.label.all[val.index]
  
  ## simple DNN as a benchmark
  set_random_seed(1)
  simple.model <- keras_model_sequential()
  
  simple.model %>%
    layer_dense(units = 2, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  simple.model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 0.005),
    loss = 'mse',
    metrics = list('mse')
  )
  
  dnn_simple_history = simple.model %>% fit(
    data.input.train,
    data.label.train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0
  )
  
  # model prediction
  train.pred = as.numeric(simple.model %>% predict(data.input.train))
  val.pred = as.numeric(simple.model %>% predict(data.input.val))

  test.out.temp = rbind(
  c(f.func.com(1, 1, 2, data.input.train, data.label.train,
             data.input.val, data.label.val), 4),
  c(f.func.com(1, 1, 3, data.input.train, data.label.train,
             data.input.val, data.label.val), 5),
  c(f.func.com(1, 1, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  c(f.func.com(1, 2, 3, data.input.train, data.label.train,
             data.input.val, data.label.val), 6),
  c(f.func.com(1, 2, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  c(f.func.com(1, 3, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 8),
  
  c(f.func.com(2, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 5),
  c(f.func.com(2, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  c(f.func.com(2, 1, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  c(f.func.com(2, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  c(f.func.com(2, 2, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 8),
  c(f.func.com(2, 3, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 9),
  
  c(f.func.com(3, 1, 2, data.input.train, data.label.train,
               data.input.val, data.label.val), 6),
  c(f.func.com(3, 1, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 7),
  c(f.func.com(3, 1, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 8),
  c(f.func.com(3, 2, 3, data.input.train, data.label.train,
               data.input.val, data.label.val), 8),
  c(f.func.com(3, 2, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 9),
  c(f.func.com(3, 3, 4, data.input.train, data.label.train,
               data.input.val, data.label.val), 10),
  
  c((mean((train.pred-data.label.train)^2)),
    (mean((val.pred-data.label.val)^2)),
    count_params(simple.model)
    )
  
  )
  
  test.tab.out[,, cross.ind] = test.out.temp

}

test.out = apply(test.tab.out, 1:2, mean)
print(test.out)

#########################################################################
## final the optimal lambda
lambda.vec = seq(0, 10, length.out = 1000)
lambda.out.vec = rep(NA, length(lambda.vec))

for (lambda.itt in 1:length(lambda.vec)){
  lambda.eval = lambda.vec[lambda.itt]
  
  cor.1.vec = test.out[1:18,1]/DNN.train.MSE - 1 + 
    lambda.eval*test.out[1:18,3]
  cor.2.vec = test.out[1:18,2]/DNN.val.MSE - 1 + 
    lambda.eval*test.out[1:18,3]
  
  lambda.out.vec[lambda.itt] = cor(cor.1.vec, cor.2.vec, method = "spearman")
  
}

print(unique(lambda.out.vec))
# print(lambda.out.vec)

lambda.opt = lambda.vec[which.max(lambda.out.vec)]
print(lambda.opt)

print(which.min(test.out[,1]/DNN.train.MSE - 1 + lambda.opt*test.out[,3]))
print(which.min(test.out[,2]/DNN.val.MSE - 1 + lambda.opt*test.out[,3]))
print(which.min(test.out[,1]))
print(which.min(test.out[,2]))

print(cor(test.out[,1], test.out[,2], method = "spearman"))
print(cor(test.out[,1]/DNN.train.MSE - 1 + lambda.opt*test.out[,3],
          test.out[,2]/DNN.val.MSE - 1 + lambda.opt*test.out[,3],
          method = "spearman"))

###############################################
## final results
final.results.out = data.frame(test.out)
final.results.out = rbind(final.results.out,
                          c(round(DNN.train.MSE, 4),
                            round(DNN.val.MSE, 4), count_params(model)))
colnames(final.results.out) = c("MSE.train", "MSE.val", "p")
final.results.out$cp.train = final.results.out$MSE.train/DNN.train.MSE - 1 + 
  lambda.opt*final.results.out$p
final.results.out$cp.val = final.results.out$MSE.val/DNN.val.MSE - 1 + 
  lambda.opt*final.results.out$p

final.results.out = final.results.out[,c(3, 1, 2, 4, 5)]

## latex table 

# final.results.out = round(final.results.out, 5)
# final.results.out = rbind(final.results.out, 
#                           c(count_params(model), round(DNN.train.MSE, 5), 
#                             round(DNN.val.MSE, 5), "", ""))

print(xtable(final.results.out, digits = c(0, 0, 5, 5, 1, 1)), 
      include.rownames = TRUE)

## final model
#####################
## get the final model
set.seed(1)
opt.par.init = rnorm(7, 0, 0.3)
opt.par = optimg(opt.par.init, f.func.mse,
                 l2.ind.mse.in = 2, 
                 l1.1.ind.mse.in = 2,
                 l1.2.ind.mse.in = 3,
                 x.total.mse.in = data.input.all, 
                 y.total.mse.in = data.label.all,
                 method="ADAM")
print(opt.par$par)

x1 = data.input.all[,1]
x2 = data.input.all[,2]
x3 = data.input.all[,3]

y = (-0.009760092)*(((qnorm(1-x2)+qnorm(1-x3)+1.635341486)/x1)^2*(-0.095984247)) + 
  0.089427012*(((qnorm(1-x2)+qnorm(1-x3)-0.492418714)/x1+
                  1.396547524)^2*(0.141250215))
print(mean((y-data.label.all)^2))

y.new = (0.056*((qnorm(1-x2)+qnorm(1-x3)+1.64)/x1)^2+
           0.758*((qnorm(1-x2)+qnorm(1-x3)-0.49)/x1+1.40)^2)/60

y.raw = 60*y