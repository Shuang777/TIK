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
from nnet_trainer import Nnet
from data_generator import DataGenerator

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

if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 6:
    raise TypeError ('USAGE: run_tf.py data_tr ali_tr gmm_dir dnn_dir')

config_file = sys.argv[1]
data        = sys.argv[2]
ali_dir      = sys.argv[3]
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
shutil.copyfile(config_file, exp+'/config')
config.read(config_file)

# prepare data
Popen(['utils/subset_data_dir_tr_cv.sh', data, exp+'/tr90', exp+'/cv10']).communicate()

## Generate pdf indices
p1 = Popen (['ali-to-pdf', '%s/final.mdl' % exp, 'ark:gunzip -c %s/ali.*.gz |' % ali_dir,
             'ark,t:-'], stdout=PIPE)
ali_labels = {}
for line in p1.stdout:
  line = line.split()
  utt = line[0].decode(sys.stdout.encoding)
  ali_labels[utt] = [int(i) for i in line[1:]]

tr_gen = DataGenerator (exp+'/tr90', ali_labels, ali_dir, exp, 'train', config.items('nnet'), shuffle=True)
cv_gen = DataGenerator (exp+'/cv10', ali_labels, ali_dir, exp, 'cv', config.items('nnet'))

# get the feature input dim
input_dim = tr_gen.getFeatDim()
output_dim = get_model_pdfs(gmm)

# save alignment priors
tr_gen.save_target_counts(output_dim, exp+'/ali_train_pdf.counts')

# save input_dim and output_dim
open(exp+'/input_dim', 'w').write(str(input_dim))
open(exp+'/output_dim', 'w').write(str(output_dim))

# set gpu ID
p1 = Popen ('pick-gpu', stdout=PIPE)
gpu_id = int(p1.stdout.read())
if gpu_id == -1:
    raise RuntimeError("Unable to pick gpu")
logger.info("Selecting gpu %d", gpu_id)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# create the neural net
nnet_conf = config.items('nnet')
optimizer_conf = config.items('optimizer')
input_dim = tr_gen.getFeatDim()

nnet = Nnet(nnet_conf, optimizer_conf, input_dim, output_dim)
mlp_init = exp+'/model.init'

if os.path.isfile(exp+'/.mlp_best'):
    mlp_best = open(exp+'/.mlp_best').read()
    logger.info("loading model from %s", mlp_best)
    nnet.read(mlp_best)
elif os.path.isfile(mlp_init+'.index'):
    logger.info("loading model from %s", mlp_init)
    nnet.read(mlp_init)
    mlp_best = mlp_init
else:
    nnet.init_nnet()
    logger.info("initialize model to %s", mlp_init)
    nnet.write(mlp_init)
    mlp_best = mlp_init

# get all variables for nnet training
initial_lr = config.getfloat('nnet', 'initial_learning_rate')
keep_lr_iters = config.getint('nnet', 'keep_lr_iters')
min_iters = config.getint('nnet', 'min_iters')
max_iters = config.getint('nnet', 'max_iters')
halving_factor = config.getfloat('nnet', 'halving_factor')
start_halving_impr = config.getfloat('nnet', 'start_halving_impr')
end_halving_impr = config.getfloat('nnet', 'end_halving_impr')

current_lr = initial_lr
if os.path.isfile(exp+'/.learn_rate'):
    current_lr = float(open(exp+'/.learn_rate').read())
if os.path.isfile(exp+'/.halving'):
    halving = bool(open(exp+'/.halving').read())
else:
    halving = False

logger.info("### neural net training started at %s", datetime.datetime.today())

loss = nnet.test(exp+'/log/initial.log', cv_gen)
logger.info("ITERATION 0: loss on cv %.3f", loss)

for i in range(max_iters):
    log_info = "ITERATION %d:" % (i+1)

    mlp_current_base = "model_iter%d" % (i+1)

    if os.path.isfile(exp+'/.done_iter'+str(i+1)):
        iter_model = match_iter_model(exp+'/nnet', mlp_current_base)
        logger.info("%s skipping... %s trained", log_info, iter_model)
        continue

    loss_tr = nnet.train(exp+'/log/iter'+str(i+1)+'.tr.log', tr_gen, current_lr)
    loss_new = nnet.test(exp+'/log/iter'+str(i+1)+'.cv.log', cv_gen)

    loss_prev = loss

    mlp_current = "%s/nnet/%s_lr%f_tr%.3f_cv%.3f" % (exp, mlp_current_base, current_lr, loss_tr, loss_new)

    if loss_new < loss or i < keep_lr_iters or i < min_iters:
        # accepting: the loss was better or we have fixed learn-rate
        loss = loss_new
        mlp_best = mlp_current
        nnet.write(mlp_best)
        logger.info("%s nnet accepted %s", log_info, mlp_best.split('/')[-1])
        open(exp + '/.mlp_best', 'w').write(mlp_best)
    else:
        mlp_rej = mlp_current + "_rejected"
        nnet.write(mlp_rej)
        logger.info("%s nnet rejected %s", log_info, mlp_rej.split('/')[-1])

    open(exp + '/.done_iter'+str(i+1), 'w').write("")
    
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

