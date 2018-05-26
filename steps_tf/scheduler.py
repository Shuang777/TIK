import os
import fnmatch
import logging
import datetime

def match_iter_model(directory, model_base):
  for file in os.listdir(directory):
    if fnmatch.fnmatch(file, model_base+'*') and file.endswith(".index"):
      return file

def run_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, tr_gen, cv_gen, 
                  nnet_train_conf, nnet_valid_conf):

  scheduler_type = scheduler_conf.get('scheduler_type', 'newbob')

  if scheduler_type == 'newbob':
    mlp_best = newbob_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, 
                                tr_gen, cv_gen, nnet_train_conf, nnet_valid_conf)
  elif scheduler_type == 'exponential':
    mlp_best = exponential_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, 
                                     tr_gen, cv_gen, nnet_train_conf, nnet_valid_conf)
  else:
    raise RuntimeError("scheduler type %s not supported" % scheduler_type)

  return mlp_best


def exponential_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, 
                          tr_gen, cv_gen, nnet_train_conf, nnet_valid_conf):

  initial_lr = scheduler_conf.get('initial_learning_rate', 1.0)
  final_lr = scheduler_conf.get('final_learning_rate', 0)
  num_iters = scheduler_conf.get('num_iters')

  logger.info("### neural net training started at %s", datetime.datetime.today())

  if not os.path.isfile(exp+'/.done_iter01'):
    loss, acc = nnet.iter_data(exp+'/log/iter00.cv.log', cv_gen, nnet_valid_conf, 
                               validation_mode = True)
    logger.info("ITERATION 0: loss on cv %.3f, acc_cv %s", loss, acc)

  for i in range(num_iters):
    log_info = "ITERATION %d:" % (i+1)
    current_lr = initial_lr * (final_lr / initial_lr) ** (1.0 * i / (num_iters-1))

    mlp_current_base = "model_iter%02d" % (i+1)

    if os.path.isfile(exp+'/.done_iter%02d'%(i+1)):
      iter_model = match_iter_model(exp+'/nnet', mlp_current_base)
      logger.info("%s skipping... %s trained", log_info, iter_model)
      continue

    nnet_train_conf.update({'learning_rate': current_lr})
    loss_tr, acc_tr = nnet.iter_data(exp+'/log/iter%02d.tr.log'%(i+1), tr_gen, nnet_train_conf)
    loss_cv, acc_cv = nnet.iter_data(exp+'/log/iter%02d.cv.log'%(i+1), cv_gen, nnet_valid_conf,
                                     validation_mode = True)

    mlp_best = "%s/nnet/%s_lr%f_tr%.3f_cv%.3f" % (exp, mlp_current_base, current_lr, loss_tr, loss_cv)

    nnet.write(mlp_best)
    open(exp+'/nnet/iter%02d.model.txt'%(i+1), 'w').write(mlp_best)
    logger.info("%s done %s, acc_tr %s, acc_cv %s", log_info, mlp_best.split('/')[-1], acc_tr, acc_cv)

    open(exp + '/.done_iter%02d'%(i+1), 'w').write("")
  
  # end of train loop
  return mlp_best
  

def newbob_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, 
                     tr_gen, cv_gen, nnet_train_conf, nnet_valid_conf):
  
  initial_lr = scheduler_conf.get('initial_learning_rate', 1.0)
  keep_lr_iters = scheduler_conf.get('keep_lr_iters', 0)
  min_iters = scheduler_conf.get('min_iters')
  max_iters = scheduler_conf.get('max_iters')
  halving_factor = scheduler_conf.get('halving_factor')
  start_halving_impr = scheduler_conf.get('start_halving_impr')
  end_halving_impr = scheduler_conf.get('end_halving_impr')

  current_lr = initial_lr
  if os.path.isfile(exp+'/.learn_rate'):
    current_lr = float(open(exp+'/.learn_rate').read())
  if os.path.isfile(exp+'/.halving'):
    halving = bool(open(exp+'/.halving').read())
  else:
    halving = False

  logger.info("### neural net training started at %s", datetime.datetime.today())

  loss, acc = nnet.iter_data(exp+'/log/iter00.cv.log', cv_gen, nnet_valid_conf, validation_mode = True)
  logger.info("ITERATION 0: loss on cv %.3f, acc_cv %s", loss, acc)

  for i in range(max_iters):
    log_info = "ITERATION %d:" % (i+1) 

    mlp_current_base = "model_iter%02d" % (i+1)

    if os.path.isfile(exp+'/.done_iter%02d'%(i+1)):
      iter_model = match_iter_model(exp+'/nnet', mlp_current_base)
      logger.info("%s skipping... %s trained", log_info, iter_model)
      continue

    nnet_train_conf.update({'learning_rate': current_lr})
    loss_tr, acc_tr = nnet.iter_data(exp+'/log/iter%02d.tr.log'%(i+1), tr_gen, nnet_train_conf)
    loss_cv, acc_cv = nnet.iter_data(exp+'/log/iter%02d.cv.log'%(i+1), cv_gen, nnet_valid_conf, 
                                     validation_mode = True)
    loss_prev = loss
    mlp_current = "%s/nnet/%s_lr%f_tr%.3f_cv%.3f" % \
                  (exp, mlp_current_base, current_lr, loss_tr, loss_cv)

    if loss_cv < loss or i < keep_lr_iters or i < min_iters:
      # accepting: the loss was better or we have fixed learn-rate
      loss = loss_cv
      mlp_best = mlp_current
      nnet.write(mlp_best)
      open(exp+'/nnet/iter%02d.model.txt'%(i+1), 'w').write(mlp_best)
      logger.info("%s nnet accepted %s, acc_tr %s, acc_cv %s", log_info, mlp_best.split('/')[-1], acc_tr, acc_cv)
      open(exp + '/.mlp_best', 'w').write(mlp_best)
    else:
      mlp_rej = mlp_current + "_rejected"
      nnet.write(mlp_rej)
      open(exp+'/nnet/iter%02d.model.txt'%(i+1), 'w').write(mlp_rej)
      logger.info("%s nnet rejected %s, acc_tr %s, acc_cv %s", log_info, mlp_rej.split('/')[-1], acc_tr, acc_cv)
      nnet.read(mlp_best)
      nnet.init_training(optimizer_conf)

    open(exp + '/.done_iter%02d'%(i+1), 'w').write("")
    
    if i < keep_lr_iters:
      continue
    
    # stopping criterion
    rel_impr = (loss_prev - loss) / loss_prev
    if halving and rel_impr < end_halving_impr:
      if i < min_iters:
        logger.info("we were supposed to finish, but we continue as min_iters: %d", min_iters)
        continue
      logger.info("finished, too small rel. improvement %.3f", rel_impr)
      break

    if rel_impr < start_halving_impr:
      halving = True
      open(exp+'/.halving', 'w').write(str(halving))

    if halving:
      current_lr = current_lr * halving_factor
      open(exp+'/.learn_rate', 'w').write(str(current_lr))

  # end of train loop
  return mlp_best

