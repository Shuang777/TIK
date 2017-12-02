#!/usr/bin/python3

import os
import sys
import re
import shutil
import datetime
import logging
import fnmatch
from six.moves import configparser
from subprocess import Popen, PIPE
from nnet_trainer import NNTrainer
from data_generator import DataGenerator
import section_config   # my own config parser after configparser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

#get number of output labels
def get_model_pdfs (gmm):
  p1 = Popen (['hmm-info', '%s/final.mdl' % gmm], stdout=PIPE)
  for line in p1.stdout.read().splitlines():
    if 'pdfs' in str(line):
      return int(line.split()[-1])

def match_iter_model(directory, model_base):
  for file in os.listdir(directory):
    if fnmatch.fnmatch(file, model_base+'*') and not file.endswith(".meta"):
      return file

def get_alignments(exp, ali_dir):
  p1 = Popen (['ali-to-pdf', '%s/final.mdl' % exp, 'ark:gunzip -c %s/ali.*.gz |' % ali_dir,
               'ark,t:-'], stdout=PIPE)
  ali_labels = {}
  for line in p1.stdout:
    line = line.split()
    utt = line[0].decode(sys.stdout.encoding)
    ali_labels[utt] = [int(i) for i in line[1:]]
  return ali_labels


if __name__ != '__main__':
  raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 6:
  raise TypeError ('USAGE: run_tf.py data_tr ali_tr gmm_dir dnn_dir')

config_file = sys.argv[1]
data        = sys.argv[2]
ali_dir     = sys.argv[3]
gmm         = sys.argv[4]
exp         = sys.argv[5]

# prepare data dir
os.path.isdir(exp) or os.makedirs (exp)
os.path.isdir(exp+'/log') or os.makedirs (exp+'/log')
os.path.isdir(exp+'/nnet') or os.makedirs (exp+'/nnet')

# log to file
logger.addHandler(logging.FileHandler(exp+'/log/train.log', mode = 'w'))
logger.info("### command line: %s", " ".join(sys.argv))

# copy necessary files
shutil.copyfile(gmm+'/tree', exp+'/tree')
shutil.copyfile(gmm+'/final.mat', exp+'/final.mat')
shutil.copyfile(gmm+'/tree', exp+'/tree')
Popen (['copy-transition-model', '--binary=false', gmm+'/final.mdl' ,exp+'/final.mdl']).communicate()

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
nnet_conf = section_config.parse(config.items('nnet'))
nnet_train_conf = section_config.parse(config.items('nnet-train'))
optimizer_conf = section_config.parse(config.items('optimizer'))
scheduler_conf = section_config.parse(config.items('scheduler'))
feature_conf = section_config.parse(config.items('feature'))

nnet_proto_file = config.get('general', 'nnet_proto', fallback = None)
summary_dir = config.get('general', 'summary_dir', fallback = None)
summary_dir = exp + '/' + summary_dir if summary_dir is not None else None

# separate data into 10% cv and 90% training
Popen(['utils/subset_data_dir_tr_cv.sh', '--cv-spk-percent', '10', data, exp+'/tr90', exp+'/cv10']).communicate()

# Generate pdf indices
ali_labels = get_alignments(exp, ali_dir)

# prepare training data generator
if nnet_conf['nnet_arch'] == 'lstm':
  data_gen_type = 'utterance'
elif nnet_conf['nnet_arch'] == 'dnn':
  data_gen_type = 'frame'

tr_gen = DataGenerator (data_gen_type, exp+'/tr90', ali_labels, ali_dir, 
                        exp, 'train', feature_conf, shuffle=True)

cv_gen = DataGenerator (data_gen_type, exp+'/cv10', ali_labels, ali_dir, 
                        exp, 'cv', feature_conf)

# get the feature input dim
input_dim = tr_gen.get_feat_dim()
output_dim = get_model_pdfs(gmm)
max_length = feature_conf.get('max_length', None)
sliding_window = feature_conf.get('sliding_window', None)
jitter_window = feature_conf.get('jitter_window', None)

# save alignment priors
tr_gen.save_target_counts(output_dim, exp+'/ali_train_pdf.counts')

# save input_dim and output_dim
open(exp+'/input_dim', 'w').write(str(input_dim))
open(exp+'/output_dim', 'w').write(str(output_dim))
if max_length is not None:
  open(exp+'/max_length', 'w').write(str(max_length))
if sliding_window is not None:
  open(exp+'/sliding_window', 'w').write(str(sliding_window))
if jitter_window is not None:
  open(exp+'/jitter_window', 'w').write(str(jitter_window))

if 'init_file' in scheduler_conf:
  logger.info("Initializing graph using %s", scheduler_conf['init_file'])

nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, 
                 feature_conf['batch_size'], summary_dir = summary_dir,
                 max_length = max_length)

mlp_init = exp+'/model.init'

if os.path.isfile(exp+'/.mlp_best'):
  mlp_best = open(exp+'/.mlp_best').read()
  logger.info("loading model from %s", mlp_best)
  nnet.read(mlp_best)
elif os.path.isfile(mlp_init+'.index') and 'init_file' not in nnet_conf:
  logger.info("loading model from %s", mlp_init)
  nnet.read(mlp_init)
  mlp_best = mlp_init
else:
  if nnet_proto_file is None:
    nnet_proto_file = exp+'/nnet.proto'
    nnet.make_proto(nnet_conf, nnet_proto_file)
  else:
    shutil.copyfile(nnet_proto_file, exp+'/nnet.proto')

  nnet.init_nnet(nnet_proto_file, init_file = scheduler_conf.get('init_file', None))

  logger.info("initialize model to %s", mlp_init)
  nnet.write(mlp_init)
  mlp_best = mlp_init

nnet.init_training(optimizer_conf)

# get all variables for nnet training
initial_lr = scheduler_conf.get('initial_learning_rate', 1.0)
final_lr = scheduler_conf.get('final_learning_rate', 0)
num_iters = scheduler_conf.get('num_iters')

logger.info("### neural net training started at %s", datetime.datetime.today())

if not os.path.isfile(exp+'/.done_iter01'):
  loss, acc = nnet.iter_data(exp+'/log/iter00.cv.log', cv_gen, None)
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

  loss_cv, acc_cv = nnet.iter_data(exp+'/log/iter%02d.cv.log'%(i+1), cv_gen, None)

  mlp_best = "%s/nnet/%s_lr%f_tr%.3f_cv%.3f" % (exp, mlp_current_base, current_lr, loss_tr, loss_cv)

  nnet.write(mlp_best)
  open(exp+'/nnet/iter%02d.model.txt'%(i+1), 'w').write(mlp_best)
  logger.info("%s done %s, acc_tr %s, acc_cv %s", log_info, mlp_best.split('/')[-1], acc_tr, acc_cv)

  open(exp + '/.done_iter%02d'%(i+1), 'w').write("")
  
# end of train loop

if mlp_best != mlp_init:
  open(exp+'/final.model.txt', 'w').write(mlp_best)
  logger.info("Succeed training the neural network in %s", exp)
  logger.info("### training complete at %s", datetime.datetime.today())
else:
  raise RuntimeError("Error training neural network...")

# End

