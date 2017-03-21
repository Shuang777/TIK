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
from signal import signal, SIGPIPE, SIG_DFL
from nnet_trainer import NNTrainer
import section_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

if __name__ != '__main__':
  raise ImportError ('This script can only be run, and can\'t be imported')

logger.info(" ".join(sys.argv))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--use-gpu', type = str)
arg_parser.add_argument('--sleep', type = int)
arg_parser.add_argument('config_file', type = str)
arg_parser.add_argument('model_file', type = str)
arg_parser.add_argument('prior_counts_file', type = str)
args = arg_parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

feature_conf = section_config.parse(config.items('feature'))
nnet_conf = section_config.parse(config.items('nnet'))
optimizer_conf = section_config.parse(config.items('optimizer'))

srcdir = os.path.dirname(args.model_file)

input_dim = int(open(srcdir+'/input_dim').read())
output_dim = int(open(srcdir+'/output_dim').read())
splice = feature_conf['context_width']

if os.path.isfile(srcdir+'/max_length'):
  max_length = int(open(srcdir+'/max_length').read())
else:
  max_length = None

# set gpu
logger.info("use-gpu: %s", str(args.use_gpu))

logger.info("initializing the graph")
nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, 
                 feature_conf['batch_size'], use_gpu = False,
                 max_length = max_length)

if os.path.exists(srcdir + '/multi.count'):
  num_multi = int(open(srcdir + '/multi.count').read())
else:
  num_multi = 0

logger.info("loading the model %s", args.model_file)
model_name=open(args.model_file, 'r').read()
nnet.read(model_name, num_multi = num_multi)

prior_counts = np.genfromtxt (args.prior_counts_file)
priors = prior_counts / prior_counts.sum()
log_priors = np.log(priors)

ark_in = sys.stdin.buffer
#ark_in = open('stdin','r')
ark_out = sys.stdout.buffer
encoding = sys.stdout.encoding
signal (SIGPIPE, SIG_DFL)

p1 = Popen(['splice-feats', '--print-args=false', '--left-context='+str(splice), 
            '--right-context='+str(splice), 'ark:-', 'ark:-'], stdin=ark_in, stdout=PIPE, stderr=DEVNULL)
p2 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', srcdir+'/cmvn.mat', 
             'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE, stderr=DEVNULL)

while True:
  uid, feats = kaldi_IO.read_utterance(p2.stdout)
  if uid == None:
    # we are done
    break

  log_post = nnet.predict (feats, take_log = False)
  log_likes = log_post - log_priors
  kaldi_IO.write_utterance(uid, log_likes, ark_out, encoding)

p1.stdout.close
