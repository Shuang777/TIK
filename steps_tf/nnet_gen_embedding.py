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
from nnet_queue import NNSeqQueue
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
arg_parser.add_argument('--verbose', dest = 'verbose', action = 'store_true')
arg_parser.add_argument('--gpu-id', type = int, default = -1)
arg_parser.add_argument('data', type = str)
arg_parser.add_argument('model_file', type = str)
arg_parser.add_argument('wspecifier', type = str)
arg_parser.set_defaults(use_gpu = False, verbose = False)
args = arg_parser.parse_args()

srcdir = os.path.dirname(args.model_file)

config = configparser.ConfigParser()
config.read(srcdir+'/config')

feature_conf = section_config.parse(config.items('feature'))
nnet_conf = section_config.parse(config.items('nnet'))
nnet_train_conf = section_config.parse(config.items('nnet-train'))

input_dim = int(open(srcdir+'/input_dim').read())
output_dim = parse_output_dim(srcdir+'/output_dim')
embedding_index = int(open(srcdir+'/embedding_index').read())
splice = feature_conf['context_width']

# prepare feature pipeline
feat_type = feature_conf.get('feat_type', 'raw')
cmvn_type = feature_conf.get('cmvn_type', 'utt')
delta_opts = feature_conf.get('delta_opts', '')

# set gpu, this is used to load the graph successfully, not really for gpu
logger.info("use-gpu: %s", str(args.use_gpu))
num_gpus = nnet_train_conf.get('num_gpus', 1)

logger.info("initializing the graph")
nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, 
                 feature_conf, num_gpus = num_gpus, use_gpu = args.use_gpu,
                 gpu_id = args.gpu_id)

logger.info("loading the model %s", args.model_file)
model_name=open(args.model_file, 'r').read()
nnet.read(model_name)

# here we are doing context window and feature normalization
if cmvn_type == 'utt':
  feats = 'ark:apply-cmvn --utt2spk=ark:'+args.data+'/utt2spk scp:'+args.data+'/cmvn.scp ' \
          + 'scp:'+args.data+'/feats.scp ark:- |'
elif cmvn_type == 'sliding':
  feats = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:' \
          + args.data + '/feats.scp ark:- |'
else:
  raise RuntimeError("cmvn_type %s not supported" % cmvn_type)

feats += " select-voiced-frames ark:- scp,s,cs:"+args.data+"/vad.scp ark:- |"

if feat_type == 'delta':
  feats += " add-deltas --print-args=false " + delta_opts + " ark:- ark:- |"

feats += " splice-feats --print-args=false --left-context="+str(splice) + \
         " --right-context="+str(splice)+" ark:- ark:-|"

feats += " apply-cmvn --print-args=false --norm-vars=true "+srcdir+"/cmvn.mat ark:- ark:- |"

count = 0
reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)
writer = kaldi_io.BaseFloatVectorWriter(args.wspecifier)

nnet_queue = NNSeqQueue(nnet, writer, embedding_index = embedding_index)

for uid, feats in reader:

  nnet_queue.add2queue(uid, feats)

  count += 1
  if args.verbose and count % 100 == 0:
    logger.info("LOG (nnet_gen_embedding.py) %d utterances processed" % count)

nnet_queue.close()
