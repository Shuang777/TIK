import os
import sys
import re
import shutil
import datetime
import logging
import fnmatch
import atexit
from six.moves import configparser
from subprocess import Popen, PIPE, check_output
from nnet_trainer import NNTrainer
from data_generator import FrameDataGenerator, UttDataGenerator
import section_config   # my own config parser after configparser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def match_iter_model(directory, model_base):
  for file in os.listdir(directory):
    if fnmatch.fnmatch(file, model_base+'*') and file.endswith(".index"):
      return file

def get_alignments(exp, ali_dir):
  p1 = Popen (['ali-to-pdf', '%s/final.mdl' % exp, 'ark:gunzip -c %s/ali.*.gz |' % ali_dir,
               'ark,t:-'], stdout = PIPE)
  ali_labels = {}
  for line in p1.stdout:
    line = line.split()
    utt = line[0]
    ali_labels[utt] = [int(i) for i in line[1:]]
  return ali_labels


if __name__ != '__main__':
  raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 5:
  raise TypeError ('USAGE: run_tf.py data_tr ali_tr dnn_dir')

config_file = sys.argv[1]
data        = sys.argv[2]
ali_dir     = sys.argv[3]
exp         = sys.argv[4]

# prepare data dir
os.path.isdir(exp) or os.makedirs (exp)
os.path.isdir(exp+'/log') or os.makedirs (exp+'/log')
os.path.isdir(exp+'/nnet') or os.makedirs (exp+'/nnet')

# copy necessary files
if os.path.exists(ali_dir+'/final.mat'):
  shutil.copyfile(ali_dir+'/final.mat', exp+'/final.mat')
shutil.copyfile(ali_dir+'/tree', exp+'/tree')
Popen (['copy-transition-model', ali_dir+'/final.mdl', exp+'/final.mdl']).communicate()

# read config file
config = configparser.ConfigParser()
if os.path.isfile(exp+'/config'):
  logger.info("Loading config file from original exp directory")
  config_file = exp+'/config'
else:
  logger.info("Copying config file to exp directory")
  shutil.copyfile(config_file, exp+'/config')
config.read(config_file)

# parse config sections
general_conf = section_config.parse(config.items('general'))
nnet_conf = section_config.parse(config.items('nnet'))
nnet_train_conf = section_config.parse(config.items('nnet-train'))
optimizer_conf = section_config.parse(config.items('optimizer'))
scheduler_conf = section_config.parse(config.items('scheduler'))
feature_conf = section_config.parse(config.items('feature'))

nnet_proto_file = general_conf.get('nnet_proto', None)
summary_dir = general_conf.get('summary_dir', None)
summary_dir = exp + '/' + summary_dir if summary_dir is not None else None

# separate data into 10% cv and 90% training
Popen(['utils/subset_data_dir_tr_cv.sh', '--cv-spk-percent', '10', data, exp+'/tr90', exp+'/cv10']).communicate()

# Generate pdf indices
ali_labels = get_alignments(exp, ali_dir)

num_gpus = nnet_train_conf.get('num_gpus', 1)

# prepare training data generator
if nnet_conf['nnet_arch'] == 'lstm':
  tr_gen = UttDataGenerator (exp+'/tr90', ali_labels, ali_dir, 
                          exp, 'train', feature_conf, shuffle=True, num_gpus = num_gpus)

  cv_gen = UttDataGenerator (exp+'/cv10', ali_labels, ali_dir, 
                          exp, 'cv', feature_conf, num_gpus = num_gpus)
elif nnet_conf['nnet_arch'] in ['dnn', 'bn']:
  tr_gen = FrameDataGenerator (exp+'/tr90', ali_labels, ali_dir, 
                          exp, 'train', feature_conf, shuffle=True, num_gpus = num_gpus)

  cv_gen = FrameDataGenerator (exp+'/cv10', ali_labels, ali_dir, 
                          exp, 'cv', feature_conf, num_gpus = num_gpus)

else:
  raise RuntimeError("nnet_arch %s not supported yet", nnet_conf['nnet_arch'])

atexit.register(tr_gen.clean)
atexit.register(cv_gen.clean)

# get the feature input dim
input_dim = tr_gen.get_feat_dim()
output_dim = int(check_output(
               'tree-info --print-args=false %s/tree | grep num-pdfs | awk \'{print $2}\''
               % ali_dir, shell=True).strip())

# save alignment priors
tr_gen.save_target_counts(output_dim, exp+'/ali_train_pdf.counts')

# save input_dim and output_dim
open(exp+'/input_dim', 'w').write(str(input_dim))
open(exp+'/output_dim', 'w').write(str(output_dim))

nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, feature_conf, 
                 num_gpus = num_gpus, summary_dir = summary_dir)

mlp_init = exp+'/model.init'

if os.path.isfile(exp+'/.mlp_best'):
  mlp_best = open(exp+'/.mlp_best').read().strip()
  logger.info("loading model from %s", mlp_best)
  nnet.read(mlp_best)
elif os.path.isfile(mlp_init+'.index'):
  logger.info("loading model from %s", mlp_init)
  nnet.read(mlp_init)
  mlp_best = mlp_init
else:
  if nnet_proto_file is None:
    nnet_proto_file = exp+'/nnet.proto'
    nnet.make_proto(nnet_conf, nnet_proto_file)
  else:
    shutil.copyfile(nnet_proto_file, exp+'/nnet.proto')

  nnet.init_nnet(nnet_proto_file)

  logger.info("initialize model to %s", mlp_init)
  nnet.write(mlp_init)
  mlp_best = mlp_init

nnet.init_training(optimizer_conf)

# get all variables for nnet training
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

loss, acc = nnet.iter_data(exp+'/log/iter00.cv.log', cv_gen, keep_acc = True)
logger.info("ITERATION 0: loss on cv %.3f, acc_cv %s", loss, acc)

for i in range(max_iters):
  log_info = "ITERATION %d:" % (i+1) 

  mlp_current_base = "model_iter%02d" % (i+1)

  if os.path.isfile(exp+'/.done_iter%02d'%(i+1)):
    iter_model = match_iter_model(exp+'/nnet', mlp_current_base)
    logger.info("%s skipping... %s trained", log_info, iter_model)
    continue

  loss_tr, acc_tr = nnet.iter_data(exp+'/log/iter%02d.tr.log'%(i+1), tr_gen, 
                                   learning_rate = current_lr,
                                   keep_in_prob = nnet_train_conf.get('keep_in_prob', 1.0),
                                   keep_out_prob = nnet_train_conf.get('keep_out_prob', 1.0))

  loss_cv, acc_cv = nnet.iter_data(exp+'/log/iter%02d.cv.log'%(i+1), cv_gen, 
                                   keep_acc = True)

  loss_prev = loss

  mlp_current = "%s/nnet/%s_lr%f_tr%.3f_cv%.3f" % (exp, mlp_current_base, current_lr, loss_tr, loss_cv)

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

if mlp_best != mlp_init:
  open(exp+'/final.model.txt', 'w').write(mlp_best)
  logger.info("Succeed training the neural network in %s", exp)
  logger.info("### training complete at %s", datetime.datetime.today())
else:
  raise RuntimeError("Error training neural network...")

# End

