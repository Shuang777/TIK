import os
import sys
import numpy as np
import logging
import kaldi_io
import argparse
from time import sleep
from subprocess import Popen, PIPE
from six.moves import configparser
from nnet_trainer import NNTrainer
import section_config
from utils import *

DEVNULL = open(os.devnull, 'w')

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
arg_parser.add_argument('--apply-log', dest = 'apply_log', action = 'store_true')
arg_parser.add_argument('--no-softmax', dest = 'no_softmax', action = 'store_true')
arg_parser.add_argument('--gpu-ids', type = str, default = '-1')
arg_parser.add_argument('--verbose', dest = 'verbose', action = 'store_true')
arg_parser.add_argument('data', type = str)
arg_parser.add_argument('model_file', type = str)
arg_parser.set_defaults(use_gpu = False, apply_log = False, no_softmax = False, verbose = False)
args = arg_parser.parse_args()

srcdir = os.path.dirname(args.model_file)

config = configparser.ConfigParser()
config.read(srcdir+'/config')

feature_conf = section_config.parse(config.items('feature'))
nnet_conf = section_config.parse(config.items('nnet'))
optimizer_conf = section_config.parse(config.items('optimizer'))
nnet_train_conf = section_config.parse(config.items('nnet-train'))

input_dim = int(open(srcdir+'/input_dim').read())
output_dim = parse_int_or_list(srcdir+'/output_dim')
max_length = feature_conf.get('max_length', None)
jitter_window = feature_conf.get('jitter_window', None)
splice = feature_conf['context_width']

# prepare feature pipeline
feat_type = feature_conf.get('feat_type', 'raw')
delta_opts = feature_conf.get('delta_opts', '')

if feature_conf.get('cmvn_type', 'utt') == 'utt':
  feats = 'ark:apply-cmvn --utt2spk=ark:' + args.data + '/utt2spk ' + \
        ' scp:' + args.data + '/cmvn.scp scp:' + args.data + '/feats.scp ark:- |'
elif feature_conf['cmvn_type'] == 'sliding':
  feats = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:' + \
          args.data + '/feats.scp ark:- |'
    
if feat_type == 'delta':
  feats += ' add-deltas ' + delta_opts + ' ark:- ark:- |'
elif feat_type in ['lda', 'fmllr']:
  feats += ' splice-feats ark:- ark:- | transform-feats ' + srcdir + '/final.mat ark:- ark:- |'

if feat_type == 'fmllr':
  assert os.path.exists(args.transform)
  feats += ' transform-feats --utt2spk=ark:' + args.data + '/utt2spk' + \
           ' ark:' + args.transform + ' ark:- ark:- |'

if args.apply_log and args.no_softmax:
  raise RuntimeError("Cannot use both --apply-log --no-softmax")

# set gpu
logger.info("use-gpu: %s", str(args.use_gpu))
num_gpus = nnet_train_conf.get('num_gpus', 1)

logger.info("initializing the graph")

# we have feature_conf['batch_size'] * num_gpus as batch_size because of multi-gpu training.
# but during decoding we only use at most 1 gpu
feature_conf['batch_size'] = feature_conf['batch_size'] * num_gpus
nnet = NNTrainer(nnet_conf, input_dim, output_dim, feature_conf, 
                 num_gpus = 1, use_gpu = args.use_gpu, gpu_ids = args.gpu_ids)

logger.info("loading the model %s", args.model_file)
model_name=open(args.model_file, 'r').read()
nnet.read(model_name)

if args.prior_counts is not None:
  prior_counts = np.genfromtxt (args.prior_counts)
  priors = prior_counts / prior_counts.sum()
  log_priors = np.log(priors)

# here we are doing context window and feature normalization
feats += ' splice-feats --print-args=false --left-context='+str(splice) + \
        ' --right-context='+str(splice) + ' ark:- ark:-|'
feats += ' apply-cmvn --print-args=false --norm-vars=true ' + srcdir+'/cmvn.mat ark:- ark:- |'

count = 0
reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)
writer = kaldi_io.BaseFloatMatrixWriter('ark:-')

for uid, feats in reader:
  nnet_out = nnet.predict (feats, no_softmax = args.no_softmax)
  if args.apply_log:
    nnet_out = np.log(nnet_out)

  if args.prior_counts is not None:
    log_likes = nnet_out - log_priors
    nnet_out = log_likes

  writer.write(uid, nnet_out)
  
  count += 1
  if args.verbose and count % 10 == 0:
    logger.info("LOG (nnet_forward.py) %d utterances processed" % count)

logger.info("LOG (nnet_forward.py) Total %d utterances processed" % count)
