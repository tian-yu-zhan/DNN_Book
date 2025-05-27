
library(gMCP)
library(doParallel)  
library(MASS)
library(nloptr)
library(stringr)
library(ANN2)
library(keras)
library(reticulate)
library(tensorflow)
library(keras)
library(tibble)

seed.vec = c(1, 2, 3)
table.out = matrix(NA, nrow = length(seed.vec),
                   ncol = 7)

for (seed.ind in 1:length(seed.vec)){
  
  seed.number = seed.vec[seed.ind]
  set.seed(seed.number)
  
  n.sim = 10^6
  obj.weight = c(0, 0.6, 0.2, 0.1, 0.1) 
  n.hypo = length(obj.weight) 
  n.graph = 10^3
  alpha.const = c(1, rep(0, 4)) 
  w.const = matrix(1, nrow = n.hypo, ncol = n.hypo)
  diag(w.const) = 0
  w.const[,1] = 0
  
  type.1.error = 0.025 
  pow.vec = c(0.95, 0.9, 0.85, 0.65, 0.6) 
  trt.vec = qnorm(1-type.1.error)-qnorm(pow.vec,
                                        lower.tail = FALSE) 
  sigma.mat = matrix(0.5, nrow = n.hypo, ncol = n.hypo)
  diag(sigma.mat) = 1
  
  
  sigmoid <- function(x) {
    return(1 / (1 + exp(-x)))
  }
  
  sig_der = function(x){
    return(sigmoid(x)*(1-sigmoid(x)))
  }
  
  draw.alpha.fun = function(n.hypo, n.graph,
                            alpha.const.in){
    temp = matrix(runif(n.hypo*n.graph, 0, 1),
                  nrow = n.graph, ncol = n.hypo)
    temp[, alpha.const.in==0] = 0
    temp = temp/(apply(temp, 1, sum)+0.0001)
    return(temp)
  }
  
  draw.w.fun = function(n.hypo, n.graph, w.const.in){
    temp = array(runif(n.hypo*n.hypo*n.graph, 0, 1),
                 dim = c(n.hypo, n.hypo, n.graph))
    for (i in 1:n.graph){
      temp.in = temp[,,i]
      temp.in[w.const.in==0] = 0
      temp[,,i] = temp.in
    }
    
    norm = apply(temp, 3, rowSums)
    norm = array(rep(norm, each = n.hypo),
                 dim = c(n.hypo, n.hypo, n.graph))
    norm = aperm(norm, c(2,1,3))+0.0001
    temp = temp/ norm
    return(temp)
  }
  
  graph.power = function(alpha.in, w.in, type.1.error.in,
                         pval.sim.mat.in){
    graph.in = matrix2graph(w.in, alpha.in)
    out_seq = graphTest(pvalues = t(pval.sim.mat.in),
                        graph = graph.in, alpha = type.1.error.in)
    out.power = apply(out_seq, 2, mean)
    return(out.power)
  }
  
  obtain.name.func = function(alpha.const.in,
                              w.const.in){
    n.hypo = length(alpha.const.in)
    name.free.space = head(paste0("a", which(!alpha.const==0)), -1)
    
    for (i in 1:dim(w.const.in)[1]){
      w.const.temp = w.const.in[i, ]
      name.free.space = c(name.free.space, head(paste0("w", i, "_",
                                                       which(!w.const.temp==0)), -1))
    }
    
    name.free.plus = paste(name.free.space,
                           collapse = "+")
    name.free.comma = paste(name.free.space,
                            collapse = ",")
    
    newlist = list("name.free.space" = name.free.space,
                   "name.free.comma" = name.free.comma,
                   "name.free.plus" = name.free.plus)
    return(newlist)
  }
  
  sim.data.function = function(n.hypo.in, n.sim.in,
                               trt.vec.in, alpha.fit.in, w.fit.in, sigma.in,
                               corr.in){
    sim.data.time = Sys.time()
    
    trt.sim.mat = t(mvrnorm(n = n.sim.in, trt.vec.in,
                            Sigma = sigma.in))
    pval.sim.mat = pnorm(trt.sim.mat, lower.tail = FALSE)
    n.graph.in = dim(alpha.fit)[1]
    data.net = cbind(alpha.fit.in, matrix(aperm(w.fit.in,
                                                c(3,2,1)), nrow =  n.graph.in,
                                          ncol = n.hypo.in*n.hypo.in))
    
    data.net = data.frame(data.net)
    colnames(data.net) = c(paste0("a", 1:n.hypo.in), 
                           paste0("w", as.vector(sapply(1:n.hypo.in,
                                                        function(x){paste0(x,"_", 1:n.hypo.in)}))))
    
    pow.vec.in = pnorm(qnorm(1-type.1.error), mean = trt.vec.in, lower.tail = FALSE)
    
    target.power.in = rep(0, n.graph.in)
    for (i in 1:n.graph.in){
      graph.power.fit = graph.power(
        as.vector(alpha.fit.in[i, ]),
        as.matrix(w.fit.in[,,i]),
        type.1.error, 
        pval.sim.mat)
      
      target.power.in[i] = sum(graph.power.fit*
                                 obj.weight)/sum(obj.weight)
    }
    
    data.net$target.power = target.power.in
    
    assump.out = matrix(NA, nrow=2,
                        ncol=length(trt.vec.in))
    assump.out[1, ] = pnorm(qnorm(1-type.1.error),
                            mean=trt.vec.in, lower.tail = FALSE)
    assump.out[2, ] = apply(pval.sim.mat, 1,
                            function(x){mean(x<=type.1.error)})
    rownames(assump.out) = c("true_power", "sim_power")
    
    data.net.all = data.net
    data.net$target.power.norm =
      (data.net$target.power-min(data.net$target.power))/
      (max(data.net$target.power)-
         min(data.net$target.power))
    data.net$target.power.norm = data.net$target.power.norm*0.4+0.3
    
    newlist = list("pval.matrix" = pval.sim.mat,
                   "data.matrix" = data.net,
                   "data.matrix.all" = data.net.all,
                   "sim.data.time.diff" = difftime(Sys.time(), sim.data.time, units="secs"))
    return(newlist)
  }
  
  neu.function = function(n.node.in, data.net.in,
                          pval.sim.mat.in, obtain.name.fit, df.fit.tol.in,
                          df.max.n.in, df.max.t.in){
    
    neu.time = Sys.time()
    name.free.space = obtain.name.fit$name.free.space
    name.free.plus = obtain.name.fit$name.free.plus
    name.free.comma = obtain.name.fit$name.free.comma
    
    n.nodes.output = matrix(NA, nrow = 1, 
                            ncol = 9 + length(name.free.space))
    
    colnames(n.nodes.output) =  c("TD_MSE", "VD_MSE", 
                                  "opt_fit_power", "opt_real_power", "opt_rank", 
                                  name.free.space,  "max_power", "hidden", "layer", 
                                  "drop_rate")
    n.nodes.output = data.frame(n.nodes.output)
    
    n.graph = dim(data.net.in)[1]
    data.train = data.net.in
    
    data.keras.train = 
      subset(data.train, select=name.free.space)
    data.keras.train =  as_tibble(data.keras.train)
    data.keras.train.scale = scale(data.keras.train) 
    
    label.keras.train = data.train$target.power.norm
    
    col_means_train = attr(data.keras.train.scale,
                           "scaled:center")
    col_stddevs_train = attr(data.keras.train.scale,
                             "scaled:scale")
    
    set_random_seed(seed.number)
    model.opt = keras_model_sequential()
    
    model.opt %>%
      layer_dense(units = n.node.in, 
                  activation = "sigmoid") %>%
      layer_dropout(rate = 0.3) %>% 
      layer_dense(units = n.node.in, 
                  activation = "sigmoid") %>%
      layer_dropout(rate = 0.3) %>%
      layer_dense(units = 1, 
                  activation = 'sigmoid')
    
    model.opt %>% compile(optimizer =
                            optimizer_rmsprop(learning_rate = 0.001),
                          loss = 'mse', metrics = list('mse'))
    
    dnn_opt_history = model.opt %>% fit(
      data.keras.train.scale,
      label.keras.train,
      verbose = 0,
      epochs = 1000,
      batch_size = 100,
      validation_split = 0
    )
    
    print(dnn_opt_history)
    
    net.train.result = model.opt %>%
      predict(data.keras.train.scale)
    
    w1.scale = get_weights(model.opt)[[1]]
    b1.scale = as.matrix(get_weights(model.opt)[[2]])
    
    w1 = t(w1.scale/matrix(rep(col_stddevs_train,
                               dim(w1.scale)[2]), 
                           nrow = dim(w1.scale)[1], ncol = dim(w1.scale)[2]))
    b1 = b1.scale - t(w1.scale)%*%
      as.matrix(col_means_train/col_stddevs_train)
    
    w2 = t(get_weights(model.opt)[[3]])
    b2 = as.matrix(get_weights(model.opt)[[4]])
    
    w3 = t(get_weights(model.opt)[[5]])
    b3 = as.matrix(get_weights(model.opt)[[6]])
    
    eval_f <- function( x ) {
      x.mat = as.matrix(c(x))
      
      w1x = (w1)%*%x.mat + b1
      sw1x = as.matrix(c(sigmoid(w1x)))
      
      w2x = (w2)%*%sw1x + b2
      sw2x = as.matrix(c(sigmoid(w2x)))
      
      w3x = (w3)%*%sw2x + b3
      sw3x =sigmoid(w3x)
      
      der_f = function(i){
        sw1x_der = as.matrix(as.vector(c((1-sigmoid(w1x))*
                                           sigmoid(w1x)))*as.vector(c(w1[, i])))
        
        w2x_der = (w2)%*%sw1x_der
        sw2x_der = as.matrix(as.vector(c(sig_der(w2x)))*
                               as.vector(c(w2x_der)))
        
        w3x_der = (w3)%*%sw2x_der
        
        out = as.numeric(sig_der(w3x)*w3x_der)
        
        return(out)
      } 
      
      return( list( 'objective' = -sw3x,
                    'gradient' = c(-der_f( 1 ),-der_f( 2 ),-der_f( 3 ),
                                   -der_f( 4 ),-der_f( 5 ),-der_f( 6 ), 
                                   -der_f( 7 ), -der_f( 8 ),-der_f( 9 ), 
                                   -der_f( 10 ),-der_f( 11 ))))
    }
    
    data.train$fit.power = as.vector(net.train.result)
    
    data.train$fit.target.power =
      (data.train$fit.power-0.3)/0.4*
      (max(data.net.in$target.power)-
         min(data.net.in$target.power))+
      min(data.net.in$target.power)
    
    set.seed(seed.number)
    x0.in = NULL
    grad.mat = NULL
    const.text = ""
    
    alpha.free.ind = head(which(!alpha.const==0), -1)
    
    if (sum(alpha.const)>1){
      const.text = paste(const.text, 
                         paste("x[", 1:length(alpha.free.ind), "]",
                               collapse = "+"), "-1,")
      
      grad.mat.temp = rep(0, length(name.free.space))
      
      grad.mat.temp[1:length(alpha.free.ind)] = 1
      grad.mat = rbind(grad.mat,  grad.mat.temp)
      
      x0.temp.in = 
        abs(rnorm(length(alpha.free.ind)+1, 0, 1))
      x0.temp.in = x0.temp.in/sum(x0.temp.in)
      x0.temp.in = x0.temp.in[1:length(alpha.free.ind)]
      
      x0.in = c(x0.in, x0.temp.in)
    }
    
    const.end = length(alpha.free.ind)
    
    for (i in 1:dim(w.const)[1]){
      w.const.temp = w.const[i, ]
      if (sum(w.const.temp)<=1) next
      
      w.free.ind = head(which(!w.const.temp==0), -1)
      
      const.text = paste(const.text, 
                         paste("x[", const.end + 1:length(w.free.ind), "]", 
                               collapse = "+"), "-1,")
      
      grad.mat.temp = rep(0, length(name.free.space))
      grad.mat.temp[const.end + 1:length(w.free.ind)] = 1
      grad.mat = rbind(grad.mat,  grad.mat.temp)
      
      x0.temp.in = abs(rnorm(length(w.free.ind)+1, 0, 1))
      x0.temp.in = x0.temp.in/sum(x0.temp.in)
      x0.temp.in = x0.temp.in[1:length(w.free.ind)]
      
      x0.in = c(x0.in, x0.temp.in)
      
      const.end = const.end + length(w.free.ind)
    }
    
    substr(const.text, str_length(const.text),
           str_length(const.text)) <- ")"
    
    const.text = paste("constr=c(", const.text)
    
    eval_g_ineq <- function( x ) {
      eval(parse(text=const.text))
      grad = grad.mat
      return(list("constraints"=constr, "jacobian"=grad))
    }
    
    lb = rep(0, length(name.free.space))
    ub = rep(1, length(name.free.space))
    
    local_opts <- list( "algorithm" = "NLOPT_LD_AUGLAG",
                        "xtol_rel" = 1.0e-5 )
    
    opts = list( "algorithm" = "NLOPT_LD_AUGLAG",
                 "xtol_rel" = 1.0e-5, "maxeval" = 10000,
                 "local_opts" = local_opts )
    
    res = nloptr(x0=x0.in,
                 eval_f=eval_f,
                 lb=lb,
                 ub=ub,
                 eval_g_ineq=eval_g_ineq,
                 opts=opts)
    
    print(res)
    
    opt.input.temp = res$solution
    
    opt.data = as.tibble(t(as.matrix(opt.input.temp)))
    opt.data.scale = scale(opt.data, 
                           center = col_means_train, 
                           scale = col_stddevs_train)
    
    opt.fit.power.temp = model.opt %>%
      predict(opt.data.scale)
    opt.fit.power.real = -gfo.func(opt.input.temp)
    
    naive.opt.fit = naive.opt.func(nloptr.func.name =
                                     "NLOPT_LN_COBYLA", 
                                   naive.tol = df.fit.tol.in, 
                                   naive.max.n = df.max.n.in,
                                   naive.max.t = df.max.t.in, 
                                   pval.sim.mat.in = pval.sim.mat.in, 
                                   x0.given = opt.input.temp)
    
    print(naive.opt.fit$naive.fit)
    
    newlist = list("opt.val" = naive.opt.fit$naive.fit,
                   "comp.time" = difftime(Sys.time(), neu.time,
                                          units="secs"))
    
    return(newlist)
  }
  
  gfo.func = function(x.gfo){
    alpha.free.ind = head(which(!alpha.const==0), -1)
    alpha.in = as.vector(rep(0, length(alpha.const)))
    
    if (sum(alpha.const)==1){
      alpha.in = alpha.const
    } else {
      alpha.in[alpha.const==1] =
        c(x.gfo[1:length(alpha.free.ind)],
          1 - sum(x.gfo[1:length(alpha.free.ind)]))
    }
    
    const.end = length(alpha.free.ind)
    w.in = matrix(0, nrow=dim(w.const)[1],
                  ncol=dim(w.const)[1])
    
    for (i in 1:dim(w.in)[1]){
      w.const.temp = w.const[i,]
      
      if (sum(w.const.temp)==1){
        w.in[i, ] = w.const.temp
      } else {
        w.free.ind = head(which(!w.const.temp==0), -1)
        w.in[i, w.const[i,]==1] =
          c(x.gfo[1:length(w.free.ind) + const.end],
            1 - sum(x.gfo[1:length(w.free.ind) + const.end]))
        const.end = const.end + length(w.free.ind)
      }
    }
    
    alpha.in = pmin(alpha.in, 1)
    alpha.in = pmax(alpha.in, 0)
    alpha.in = alpha.in / (sum(alpha.in)+10^(-6))
    
    w.in[w.in<0] = 0
    w.in[w.in>1] = 1
    w.in = t(apply(w.in, 1,
                   function(x){x/(sum(x)+10^(-6))}))
    
    graph.power.gfo = graph.power(as.vector(alpha.in), 
                                  as.matrix(w.in), type.1.error,
                                  sim.data.fit$pval.matrix)
    
    return(-sum(graph.power.gfo*obj.weight)/
             sum(obj.weight))
  }
  
  naive.opt.func = function(nloptr.func.name, 
                            naive.tol, 
                            naive.max.n,
                            naive.max.t, 
                            pval.sim.mat.in, 
                            x0.given){
    
    set.seed(seed.number)
    naive.opt.time = Sys.time()
    
    const.text = ""
    x0.start = grad.mat = NULL
    
    alpha.free.ind = head(which(!alpha.const==0), -1)
    if (sum(alpha.const)>1){
      const.text = paste(const.text, paste("x[", 1:length(alpha.free.ind), "]", collapse = "+"), "-1,")
      grad.mat.temp = rep(0, length(name.free.space))
      grad.mat.temp[1:length(alpha.free.ind)] = 1
      grad.mat = rbind(grad.mat,  grad.mat.temp)
      
      x0.start.in = runif(n=length(alpha.free.ind)+1, 0, 1)
      x0.start.in = x0.start.in/sum(x0.start.in)
      x0.start.in = x0.start.in[1:length(alpha.free.ind)]
      
      x0.start = c(x0.start, x0.start.in)
    }
    const.end = length(alpha.free.ind)
    
    for (i in 1:dim(w.const)[1]){
      w.const.temp = w.const[i, ]
      if (sum(w.const.temp)<=1) next
      
      w.free.ind = head(which(!w.const.temp==0), -1)
      const.text = paste(const.text, 
                         paste("x[", const.end + 1:length(w.free.ind), "]",
                               collapse = "+"), "-1,")
      
      grad.mat.temp = rep(0, length(name.free.space))
      grad.mat.temp[const.end + 1:length(w.free.ind)] = 1
      grad.mat = rbind(grad.mat, grad.mat.temp)
      
      x0.start.in = runif(n=length(w.free.ind)+1, 0, 1)
      x0.start.in = x0.start.in/sum(x0.start.in)
      x0.start.in = x0.start.in[1:length(w.free.ind)]
      
      x0.start = c(x0.start, x0.start.in)
      
      const.end = const.end + length(w.free.ind)
    }
    
    substr(const.text, str_length(const.text),
           str_length(const.text)) <- ")"
    const.text = paste("constr<-c(", const.text)
    
    eval_ineq <- function( x ) {
      eval(parse(text=const.text))
      grad = grad.mat
      return( list( "constraints"=constr,
                    "jacobian"=grad ) )
    }
    
    local_opts = list( "algorithm" = nloptr.func.name,
                       "xtol_rel" = naive.tol,
                       "ftol_rel" = 0,
                       "maxeval" = 100)
    
    opts = list( "algorithm" = nloptr.func.name,
                 "xtol_rel" = naive.tol,
                 "ftol_rel" = 0,
                 "maxeval" = naive.max.n,
                 "maxtime" = naive.max.t, 
                 "local_opts" = local_opts)
    
    if (is.null(x0.given)){
      x0.start.in = x0.start
    } else {
      x0.start.in = x0.given
    }
    
    res <- nloptr( x0=x0.start.in,
                   eval_f=gfo.func,
                   lb=rep(0, length(x0.start)),
                   ub=rep(1, length(x0.start)),
                   eval_g_ineq=eval_ineq,
                   opts=opts)
    
    print(res)
    test.temp = -res$objective
    naive.input.temp = res$solution
    
    newlist = list("naive.fit" = test.temp,
                   "solution" = res$solution, 
                   "time" = difftime(Sys.time(), naive.opt.time,
                                     units="secs"))
    
    return(newlist)
  }
  
  alpha.fit = 
    draw.alpha.fun(n.hypo, n.graph, alpha.const)
  w.fit = draw.w.fun(n.hypo, n.graph, w.const)
  
  obtain.name.fit = 
    obtain.name.func(alpha.const, w.const)
  name.free.space = obtain.name.fit$name.free.space
  name.free.plus = obtain.name.fit$name.free.plus
  name.free.comma = obtain.name.fit$name.free.comma  
  
  sim.data.fit = sim.data.function(n.hypo.in = n.hypo,
                                   n.sim.in = n.sim,
                                   trt.vec.in = trt.vec,
                                   alpha.fit.in = alpha.fit,
                                   w.fit.in = w.fit,
                                   sigma.in = sigma.mat,
                                   corr.in = corr)
  
  neu.func.fit = neu.function(n.node.in = 30,
                              data.net.in = sim.data.fit$data.matrix,
                              pval.sim.mat.in = sim.data.fit$pval.matrix,
                              obtain.name.fit = obtain.name.fit,
                              df.fit.tol.in = 10^(-4), 
                              df.max.n.in = 10^4, 
                              df.max.t.in = -1)
  
  DNN.total.time = sim.data.fit$sim.data.time.diff+
    neu.func.fit$comp.time
  DNN.opt.val = neu.func.fit$opt.val
  
  COB.fit = naive.opt.func(nloptr.func.name =
                             "NLOPT_LN_COBYLA",
                           naive.tol = 10^(-4),
                           naive.max.n = -1,
                           naive.max.t = DNN.total.time*1.5,
                           pval.sim.mat.in = sim.data.fit$pval.matrix,
                           x0.given = NULL)
  
  ISR.fit = naive.opt.func(nloptr.func.name = "NLOPT_GN_ISRES",
                           naive.tol = 10^(-4),
                           naive.max.n = -1,
                           naive.max.t = DNN.total.time*1.5,
                           pval.sim.mat.in = sim.data.fit$pval.matrix,
                           x0.given = NULL)
  
  table.out[seed.ind, ] = c(seed.number, 
                            DNN.opt.val, 
                            COB.fit$naive.fit, 
                            ISR.fit$naive.fit,
                            DNN.total.time,
                            COB.fit$time,
                            ISR.fit$time
  )
}






