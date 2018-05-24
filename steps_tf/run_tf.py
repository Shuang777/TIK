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
from data_generator import SeqDataGenerator, FrameDataGenerator, UttDataGenerator, JointDNNDataGenerator
import section_config   # my own config parser after configparser
from scheduler import run_scheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def load_utt2label(utt2label_file, convert_int = False):
  utt2label = {}
  labels = set()
  with open(utt2label_file) as f:
    for line in f:
      utt, label = line.strip().split()
      if convert_int:
        utt2label[utt] = int(label)
        labels.add(int(label))
      else:
        utt2label[utt] = label
        labels.add(label)
  return utt2label, labels

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

def assign_spk_label(spks):
  spk2label = {}
  for spk in spks:
    if spk not in spk2label:
      spk2label[spk] = len(spk2label)
  return spk2label

def mapspk2label(utt2spk, spk2label):
  utt2label = {}
  for utt, spk in utt2spk.iteritems():
    utt2label[utt] = spk2label[spk]
  return utt2label

if __name__ != '__main__':
  raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 5:
  raise TypeError ('USAGE: run_tf.py data_tr ali_tr dnn_dir')

config_file = sys.argv[1]
data        = sys.argv[2]
ali_dir     = sys.argv[3]   # might be empty when we don't need it
exp         = sys.argv[4]

# prepare data dir
os.path.isdir(exp) or os.makedirs (exp)
os.path.isdir(exp+'/log') or os.makedirs (exp+'/log')
os.path.isdir(exp+'/nnet') or os.makedirs (exp+'/nnet')

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

nnet_arch = nnet_conf['nnet_arch']
buckets_tr = nnet_conf.get('buckets_tr', None)    # training buckets
summary_dir = exp+'/summary'

logger.info("Loading labels from %s" % data)
if nnet_arch in ['lstm', 'bn', 'dnn']:
  # copy necessary files
  if os.path.exists(ali_dir+'/final.mat'):
    shutil.copyfile(ali_dir+'/final.mat', exp+'/final.mat')
  shutil.copyfile(ali_dir+'/tree', exp+'/tree')
  Popen (['copy-transition-model', ali_dir+'/final.mdl', exp+'/final.mdl']).communicate()
  
  # Generate pdf indices
  ali_labels = get_alignments(exp, ali_dir)
  
  output_dim = int(check_output(
                   'tree-info --print-args=false %s/tree | grep num-pdfs | awk \'{print $2}\''
                   % ali_dir, shell=True).strip())

elif nnet_arch in ['seq2class', 'jointdnn-sid']:
  utt2label_train, _ = load_utt2label(data + '/utt2label.train', convert_int = True)
  utt2label_valid, _ = load_utt2label(data + '/utt2label.valid', convert_int = True)

  output_dim = max(utt2label_train.values())+1

elif nnet_arch == 'jointdnn':
  # separate data into 10% cv and 90% training
  utt2spk_train, spks_train = load_utt2label(data+'/train/utt2spk')
  utt2spk_valid, spks_valid = load_utt2label(data+'/valid/utt2spk')

  assert(spks_train == spks_valid)
  spk2label = assign_spk_label(spks_train)

  utt2label_train = mapspk2label(utt2spk_train, spk2label)
  utt2label_valid = mapspk2label(utt2spk_valid, spk2label)

  # copy necessary files
  if os.path.exists(ali_dir+'/final.mat'):
    shutil.copyfile(ali_dir+'/final.mat', exp+'/final.mat')
  shutil.copyfile(ali_dir+'/tree', exp+'/tree')
  Popen (['copy-transition-model', ali_dir+'/final.mdl', exp+'/final.mdl']).communicate()

  # Generate pdf indices
  ali_labels = get_alignments(exp, ali_dir)

  asr_output_dim = int(check_output(
                   'tree-info --print-args=false %s/tree | grep num-pdfs | awk \'{print $2}\''
                   % ali_dir, shell=True).strip())

  sid_output_dim = max(utt2label_train.values())+1

  output_dim = (asr_output_dim, sid_output_dim)

else:
  raise RuntimeError("nnet_arch %s not supported", nnet_arch)

num_gpus = nnet_train_conf.get('num_gpus', 1)

# prepare training data generator
if nnet_arch == 'lstm':
  tr_gen = UttDataGenerator(data, ali_labels, ali_dir, 
                            exp, 'train', feature_conf, shuffle=True, num_gpus = num_gpus)
  cv_gen = UttDataGenerator(data, ali_labels, ali_dir, 
                            exp, 'valid', feature_conf, num_gpus = num_gpus)
elif nnet_arch in ['dnn', 'bn']:
  tr_gen = FrameDataGenerator(data, ali_labels, ali_dir, 
                              exp, 'train', feature_conf, shuffle=True, num_gpus = num_gpus)
  cv_gen = FrameDataGenerator(data, ali_labels, ali_dir, 
                              exp, 'valid', feature_conf, num_gpus = num_gpus)
elif nnet_arch in ['seq2class', 'jointdnn-sid']:
  tr_gen = SeqDataGenerator(data, utt2label_train, None, exp, 'train',
                            feature_conf, shuffle=True, num_gpus = num_gpus, buckets=buckets_tr)
  cv_gen = SeqDataGenerator(data, utt2label_valid, None, exp, 'valid', 
                            feature_conf, num_gpus = num_gpus, buckets=buckets_tr)
elif nnet_arch == 'jointdnn':
  tr_gen = JointDNNDataGenerator(data, utt2label_train, ali_labels, exp, 'train', 
                                 feature_conf, shuffle=True, buckets=buckets_tr)
  cv_gen = JointDNNDataGenerator(data, utt2label_valid, ali_labels, exp, 'valid', 
                                 feature_conf, buckets=buckets_tr)
else:
  raise RuntimeError("nnet_arch %s not supported yet", nnet_arch)


# get the feature input dim
input_dim = tr_gen.get_feat_dim()
max_length = feature_conf.get('max_length', None)

if nnet_arch in ['dnn', 'lstm']:
  # save alignment priors
  tr_gen.save_target_counts(output_dim, exp+'/ali_train_pdf.counts')

# save input_dim and output_dim
open(exp+'/input_dim', 'w').write(str(input_dim))
open(exp+'/output_dim', 'w').write(str(output_dim))
if max_length is not None:
  open(exp+'/max_length', 'w').write(str(max_length))

nnet = NNTrainer(nnet_conf, input_dim, output_dim, feature_conf, 
                 gpu_ids = nnet_train_conf.get('gpu_ids', '-1'),
                 num_gpus = num_gpus, summary_dir = summary_dir)

mlp_init = exp+'/model.init'

if os.path.isfile(exp+'/.mlp_best'):
  mlp_best = open(exp+'/.mlp_best').read().strip()
  nnet.read(mlp_best)
elif os.path.isfile(mlp_init+'.index'):
  nnet.read(mlp_init)
  mlp_best = mlp_init
else:
  # we need to create the model
  if nnet_arch == 'jointdnn-sid':
    # we create the model on top of existing model
    init_model = open(exp+'/init.model.txt').read().strip()
    nnet.read(init_model)
    nnet_proto_file = exp+'/nnet.proto'
    nnet.make_proto(nnet_conf, nnet_proto_file)
    nnet.edit_model(nnet_proto_file, 'finetune-sid')
  else:
    # we create it from scratch
    nnet_proto_file = exp+'/nnet.proto'
    nnet.make_proto(nnet_conf, nnet_proto_file)
    nnet.init_nnet(nnet_proto_file)

  logger.info("initialize model to %s", mlp_init)
  nnet.write(mlp_init)
  mlp_best = mlp_init

nnet.init_training(optimizer_conf)

nnet_valid_conf = {
  'alpha': nnet_train_conf.get('alpha', None),
  'beta': nnet_train_conf.get('beta', None)
}


# run the scheduler
mlp_best = run_scheduler(logger, nnet, scheduler_conf, optimizer_conf, exp, 
                         tr_gen, cv_gen, nnet_train_conf, nnet_valid_conf)

# End
if mlp_best != mlp_init:
  open(exp+'/final.model.txt', 'w').write(mlp_best)
  logger.info("Succeed training the neural network in %s", exp)
  logger.info("### training complete at %s", datetime.datetime.today())
else:
  raise RuntimeError("Error training neural network...")

