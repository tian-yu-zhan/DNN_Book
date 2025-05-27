
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
library(ggplot2)
library(plotly)

setwd("~/Dropbox/Research/AbbVie/Book/Code_Final/Chap_2_I/")

###############################################################################
set.seed(1) 
n.ind.1 = 100
alpha = 0.05
A = 25
theta.1.train.vec = seq(from = -3, to = 3, length.out = n.ind.1)
theta.2.train.vec = seq(from = -3, to = 3, length.out = n.ind.1)
theta.1.2.train.mat = expand.grid(theta.1.train.vec, theta.2.train.vec)
n.ind = n.ind.1^2
data.train.label = rep(NA, n.ind)

###############################################################################
for (ind in 1:n.ind){
  
  theta.1.train = theta.1.2.train.mat$Var1[ind]
  theta.2.train = theta.1.2.train.mat$Var2[ind]
  
  dec.out = (A*2 + (theta.1.train^2-A*cos(0.6*pi*theta.1.train))+
    (theta.2.train^2-A*cos(0.6*pi*theta.2.train)))/120
  
  data.train.label[ind] = dec.out
}

data.plot = data.frame("x1" = theta.1.2.train.mat$Var1,
                        "x2" = theta.1.2.train.mat$Var2,
                        "y" = data.train.label)

png("data_plot.png", width = 1800, height = 1600)
print(

  ggplot(data.plot, aes(x = x1, y = x2, z = y))+
    stat_contour_filled(alpha = 1) +
        scale_x_continuous(breaks=c(-3, 0, 3))+
        scale_y_continuous(breaks=c(-3, 0, 3))+
        theme_bw() +
        theme(axis.text.x = element_text(size=60),
              axis.text.y = element_text(size=60),
              axis.title.x = element_text(size=60),
              axis.title.y = element_text(size=60),
              legend.title = element_text(size=60),
              legend.text = element_text(size=40),
              legend.key.width= unit(2, 'cm') )+
    labs(fill = "y")

      
)
dev.off()

print(summary(data.plot$y))

library(plotly)
p.ori = plot_ly(
  data.plot, x= ~x1, y= ~x2, z= ~y,
  type='mesh3d', intensity = ~y,
  colors = colorRamp(c("navyblue", "slateblue2", "chartreuse3", "yellow"))
)
print(p.ori)


###########################################################################
data.train.com = data.plot[sample(x = 1:n.ind, size = n.ind, replace = FALSE),]

data.train =  as_tibble(cbind(data.train.com$x1, data.train.com$x2))
data.train.scale = as.matrix(data.train)

data.train.label.all = data.train.com$y

DNN.cross.func = function(data.train.scale.in,
                          data.train.label.in,
                          act.func.name.in,
                          n.nodes.in,
                          drop.out.rate.in,
                          learn.rate.in,
                          epoch.in,
                          batch.size.in
                          ){
  
  cross.table.out = matrix(NA, nrow = 5, ncol = 2)
  
  for (cross.ind in 1:5){
    print(cross.ind)

    val.index = (1:(n.ind/5))+(cross.ind-1)*n.ind/5
    train.index = (1:n.ind)[-val.index]
    
    set_random_seed(1)
    model <- keras_model_sequential()
    
    model %>%
      layer_dense(units = n.nodes.in, activation = act.func.name.in) %>%
      layer_dropout(rate = drop.out.rate.in) %>% 
      layer_dense(units = n.nodes.in, activation = act.func.name.in) %>%
      layer_dropout(rate = drop.out.rate.in) %>%
      layer_dense(units = n.nodes.in, activation = act.func.name.in) %>%
      layer_dropout(rate = drop.out.rate.in) %>%
      layer_dense(units = 1, activation = 'linear')
    
    model %>% compile(
      optimizer = optimizer_rmsprop(learning_rate = learn.rate.in),
      loss = 'mse',
      metrics = list('mse')
    )
    
    dnn_history = model %>% fit(
      data.train.scale.in[train.index, ],
      data.train.label.in[train.index],
      epochs = epoch.in,
      batch_size = batch.size.in,
      validation_split = 0,
      verbose = 0
    )
    
    print(dnn_history)
    
    DNN.train.pred = predict(model, data.train.scale.in[train.index, ])
    DNN.val.pred = predict(model, data.train.scale.in[val.index, ])

    train.MSE.temp = mean((DNN.train.pred-data.train.label.in[train.index])^2)
    val.MSE.temp = mean((DNN.val.pred-data.train.label.in[val.index])^2)
    
    cross.table.out[cross.ind, ] = c(train.MSE.temp, val.MSE.temp)
  }  
    
  return(apply(cross.table.out, 2, mean))
  
}

# test = DNN.cross.func(data.train.scale.in = data.train.scale,
#                                  data.train.label.in = data.train.com$y,
#                                  act.func.name.in = "sigmoid",
#                                  n.nodes.in = 60,
#                                  drop.out.rate.in = 0.2,
#                                  learn.rate.in = 0.005,
#                                  epoch.in = 500,
#                                  batch.size.in = 500)

########################################################################
## activation function
DNN.act.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.005,
                             epoch.in = 500,
                             batch.size.in = 500)

DNN.act.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                           data.train.label.in = data.train.label.all,
                           act.func.name.in = "relu",
                           n.nodes.in = 60,
                           drop.out.rate.in = 0.2,
                           learn.rate.in = 0.005,
                           epoch.in = 500,
                           batch.size.in = 500)

DNN.act.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                           data.train.label.in = data.train.label.all,
                           act.func.name.in = "tanh",
                           n.nodes.in = 60,
                           drop.out.rate.in = 0.2,
                           learn.rate.in = 0.005,
                           epoch.in = 500,
                           batch.size.in = 500)

########################################################################
## number of nodes
DNN.nodes.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                                 data.train.label.in = data.train.label.all,
                                 act.func.name.in = "sigmoid",
                                 n.nodes.in = 60,
                                 drop.out.rate.in = 0.2,
                                 learn.rate.in = 0.005,
                                 epoch.in = 500,
                                 batch.size.in = 500)

DNN.nodes.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 120,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.005,
                             epoch.in = 500,
                             batch.size.in = 500)

DNN.nodes.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 180,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.005,
                             epoch.in = 500,
                             batch.size.in = 500)

########################################################################
## dropout rate
DNN.drop.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.1,
                             learn.rate.in = 0.005,
                             epoch.in = 500,
                             batch.size.in = 500)

DNN.drop.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.2,
                            learn.rate.in = 0.005,
                            epoch.in = 500,
                            batch.size.in = 500)

DNN.drop.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.5,
                            learn.rate.in = 0.005,
                            epoch.in = 500,
                            batch.size.in = 500)

########################################################################
## learning rate
DNN.rate.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.2,
                            learn.rate.in = 0.001,
                            epoch.in = 500,
                            batch.size.in = 500)

DNN.rate.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.2,
                            learn.rate.in = 0.01,
                            epoch.in = 500,
                            batch.size.in = 500)

DNN.rate.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.2,
                            learn.rate.in = 0.1,
                            epoch.in = 500,
                            batch.size.in = 500)

########################################################################
## epochs
DNN.epoch.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                            data.train.label.in = data.train.label.all,
                            act.func.name.in = "sigmoid",
                            n.nodes.in = 60,
                            drop.out.rate.in = 0.2,
                            learn.rate.in = 0.001,
                            epoch.in = 50,
                            batch.size.in = 500)

DNN.epoch.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.001,
                             epoch.in = 500,
                             batch.size.in = 500)

DNN.epoch.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.001,
                             epoch.in = 5000,
                             batch.size.in = 500)

########################################################################
## batch size
DNN.batch.1 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.001,
                             epoch.in = 500,
                             batch.size.in = 50)

DNN.batch.2 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.001,
                             epoch.in = 500,
                             batch.size.in = 500)

DNN.batch.3 = DNN.cross.func(data.train.scale.in = data.train.scale,
                             data.train.label.in = data.train.label.all,
                             act.func.name.in = "sigmoid",
                             n.nodes.in = 60,
                             drop.out.rate.in = 0.2,
                             learn.rate.in = 0.001,
                             epoch.in = 500,
                             batch.size.in = 5000)

###################################################################
## HP choice
n.C = 10
table.C = matrix(NA, nrow = n.C, ncol = 7)

for (ind.C in 1:n.C){
  print(ind.C)
  
  set.seed(ind.C)
  n.nodes.C = round(runif(1, min = 80, max = 160))
  drop.rate.C = runif(1, min = 0, max = 0.3)
  learn.rate.C = runif(1, min = 0.001, max = 0.02)
  epoch.C = round(runif(1, min = 10^3, max = 10^4))
  batch.C = round(runif(1, min = 10, max = 100))
  
  DNN.temp = DNN.cross.func(data.train.scale.in = data.train.scale,
                               data.train.label.in = data.train.label.all,
                               act.func.name.in = "sigmoid",
                               n.nodes.in = n.nodes.C,
                               drop.out.rate.in = drop.rate.C,
                               learn.rate.in = learn.rate.C,
                               epoch.in = epoch.C,
                               batch.size.in = batch.C)
  
  table.C[ind.C, ] = c(
    n.nodes.C, drop.rate.C, learn.rate.C, epoch.C, batch.C, 
    DNN.temp)
  
}

###########################################################
## opt model
opt.index = which.min(as.numeric(table.C[,7]))

set_random_seed(1)
opt.model <- keras_model_sequential()

opt.model %>%
  layer_dense(units = as.numeric(table.C[opt.index, 1]), 
              activation = "sigmoid") %>%
  layer_dropout(rate = as.numeric(table.C[opt.index, 2])) %>% 
  layer_dense(units = as.numeric(table.C[opt.index, 1]), 
              activation = "sigmoid") %>%
  layer_dropout(rate = as.numeric(table.C[opt.index, 2])) %>%
  layer_dense(units = as.numeric(table.C[opt.index, 1]), 
              activation = "sigmoid") %>%
  layer_dropout(rate = as.numeric(table.C[opt.index, 2])) %>%
  layer_dense(units = 1, activation = 'linear')

opt.model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = as.numeric(table.C[opt.index, 3])),
  loss = 'mse',
  metrics = list('mse')
)

opt_dnn_history = opt.model %>% fit(
  data.train.scale,
  data.train.label.all,
  epochs = as.numeric(table.C[opt.index, 4]),
  batch_size = as.numeric(table.C[opt.index, 5]),
  validation_split = 0,
  verbose = 0
)

print(opt_dnn_history)

########################################################################
## opt.model.plot
DNN.opt.pred = predict(opt.model, data.train.scale)

data.opt.plot = data.frame("x1" = data.train.scale[, 1],
                       "x2" = data.train.scale[,2],
                       "y" = DNN.opt.pred)

png("data_opt_plot.png", width = 1800, height = 1600)
print(
  
  ggplot(data.opt.plot, aes(x = x1, y = x2, z = y))+
    stat_contour_filled(alpha = 1) +
    scale_x_continuous(breaks=c(-3, 0, 3))+
    scale_y_continuous(breaks=c(-3, 0, 3))+
    theme_bw() +
    theme(axis.text.x = element_text(size=60),
          axis.text.y = element_text(size=60),
          axis.title.x = element_text(size=60),
          axis.title.y = element_text(size=60),
          legend.title = element_text(size=60),
          legend.text = element_text(size=40),
          legend.key.width= unit(2, 'cm') )+
    labs(fill = "y")
  
  
)
dev.off()

p.opt = plot_ly(
  data.opt.plot, x= ~x1, y= ~x2, z= ~y,
  type='mesh3d', intensity = ~y,
  colors = colorRamp(c("navyblue", "slateblue2", "chartreuse3", "yellow"))
)
print(p.opt)

####################################################################
## latex table
library(xtable)
table.out.1 = rbind(DNN.act.1, DNN.act.2, DNN.act.3)
table.out.1 = cbind(1:3, table.out.1)
print(xtable(table.out.1, digits=c(0, 0,4,4)), include.rownames=FALSE)

table.out.2 = rbind(DNN.nodes.1, DNN.nodes.2, DNN.nodes.3)
table.out.2 = cbind(c(60, 120, 180), table.out.2)
print(xtable(table.out.2, digits=c(0, 0,4,4)), include.rownames=FALSE)

table.out.3 = rbind(DNN.drop.1, DNN.drop.2, DNN.drop.3)
table.out.3 = cbind(c(0.1, 0.2, 0.5), table.out.3)
print(xtable(table.out.3, digits=c(0, 1,4,4)), include.rownames=FALSE)

table.out.4 = rbind(DNN.rate.1, DNN.rate.2, DNN.rate.3)
table.out.4 = cbind(c(0.001, 0.01, 0.1), table.out.4)
print(xtable(table.out.4, digits=c(0, 3,4,4)), include.rownames=FALSE)

table.out.5 = rbind(DNN.epoch.1, DNN.epoch.2, DNN.epoch.3)
table.out.5 = cbind(c(50, 500, 5000), table.out.5)
print(xtable(table.out.5, digits=c(0, 0,4,4)), include.rownames=FALSE)

table.out.6 = rbind(DNN.batch.1, DNN.batch.2, DNN.batch.3)
table.out.6 = cbind(c(50, 500, 5000), table.out.6)
print(xtable(table.out.6, digits=c(0, 0,4,4)), include.rownames=FALSE)






