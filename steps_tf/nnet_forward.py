#!/usr/bin/python3

import os
import sys
import numpy as np
import logging
import kaldi_IO
import argparse
from time import sleep
from subprocess import Popen, PIPE, DEVNULL
from six.moves import configparser
from nnet_trainer import NNTrainer
import section_config


def read_int_or_none(file_name):
  if os.path.isfile(file_name):
    return int(open(file_name).read())
  else:
    return None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

if __name__ != '__main__':
  raise ImportError ('This script can only be run, and can\'t be imported')

logger.info(" ".join(sys.argv))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--use-gpu', dest = 'use_gpu', action = 'store_true')
arg_parser.add_argument('--sleep', type = int)
arg_parser.add_argument('--prior-counts', type = str, default = None)
arg_parser.add_argument('--transform', type = str, default = None)
arg_parser.add_argument('data', type = str)
arg_parser.add_argument('model_file', type = str)
arg_parser.set_defaults(use_gpu = False)
args = arg_parser.parse_args()

srcdir = os.path.dirname(args.model_file)

config = configparser.ConfigParser()
config.read(srcdir+'/config')

feature_conf = section_config.parse(config.items('feature'))
nnet_conf = section_config.parse(config.items('nnet'))
optimizer_conf = section_config.parse(config.items('optimizer'))
nnet_train_conf = section_config.parse(config.items('nnet-train'))

input_dim = int(open(srcdir+'/input_dim').read())
output_dim = int(open(srcdir+'/output_dim').read())
max_length = feature_conf.get('max_length', None)
jitter_window = feature_conf.get('jitter_window', None)
splice = feature_conf['context_width']

# prepare feature pipeline
feat_type = feature_conf.get('feat_type', 'raw')
delta_opts = feature_conf.get('delta_opts', '')
cmd = ['apply-cmvn', '--utt2spk=ark:' + args.data + '/utt2spk', 
       'scp:' + args.data + '/cmvn.scp', 
       'scp:' + args.data + '/feats.scp', 'ark:-']
    
if feat_type == 'delta':
  cmd.extend(['|', 'add-deltas', delta_opts, 'ark:-', 'ark:-'])
elif feat_type in ['lda', 'fmllr']:
  cmd.extend(['|', 'splice-feats', 'ark:-','ark:-'])
  cmd.extend(['|', 'transform-feats', srcdir + '/final.mat', 'ark:-', 'ark:-'])

if feat_type == 'fmllr':
  assert os.path.exists(args.transform)
  cmd.extend(['|', 'transform-feats','--utt2spk=ark:' + args.data + '/utt2spk',
          'ark:%s' % args.transform, 'ark:-', 'ark:-'])

#print(cmd)
feat_pipe = Popen(' '.join(cmd), shell = True, stdout=PIPE)

# set gpu
logger.info("use-gpu: %s", str(args.use_gpu))
num_gpus = nnet_train_conf.get('num_gpus', 1)

logger.info("initializing the graph")
nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, 
                 feature_conf['batch_size'] * num_gpus, num_gpus = num_gpus,
                 max_length = max_length, use_gpu = args.use_gpu,
                 jitter_window = jitter_window)

logger.info("loading the model %s", args.model_file)
model_name=open(args.model_file, 'r').read()
nnet.read(model_name)

if args.prior_counts is not None:
  prior_counts = np.genfromtxt (args.prior_counts)
  priors = prior_counts / prior_counts.sum()
  log_priors = np.log(priors)

ark_in = feat_pipe.stdout
ark_out = sys.stdout.buffer
encoding = sys.stdout.encoding

# here we are doing context window and feature normalization
p1 = Popen(['splice-feats', '--print-args=false', '--left-context='+str(splice), 
            '--right-context='+str(splice), 'ark:-', 'ark:-'], 
            stdin = ark_in, stdout=PIPE, stderr=DEVNULL)
p2 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', srcdir+'/cmvn.mat', 
             'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE, stderr=DEVNULL)

while True:
  uid, feats = kaldi_IO.read_utterance(p2.stdout)
  if uid == None:
    # we are done
    break

  log_post = nnet.predict (feats, take_log = False)
  nnet_out = log_post
  if args.prior_counts is not None:
    log_likes = log_post - log_priors
    nnet_out = log_likes

  kaldi_IO.write_utterance(uid, nnet_out, ark_out, encoding)

feat_pipe.stdout.close()
p1.stdout.close()


