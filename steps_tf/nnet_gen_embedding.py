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

DEVNULL = open(os.devnull, 'w')

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
arg_parser.add_argument('--verbose', dest = 'verbose', action = 'store_true')
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
output_dim = int(open(srcdir+'/output_dim').read())
splice = feature_conf['context_width']

# prepare feature pipeline
feat_type = feature_conf.get('feat_type', 'raw')
delta_opts = feature_conf.get('delta_opts', '')

# set gpu, this is used to load the graph successfully, not really for gpu
logger.info("use-gpu: %s", str(args.use_gpu))
num_gpus = nnet_train_conf.get('num_gpus', 1)

logger.info("initializing the graph")
nnet = NNTrainer(nnet_conf['nnet_arch'], input_dim, output_dim, 
                 feature_conf['batch_size'], num_gpus = num_gpus,
                 use_gpu = args.use_gpu)

logger.info("loading the model %s", args.model_file)
model_name=open(args.model_file, 'r').read()
nnet.read(model_name)

# here we are doing context window and feature normalization
feats = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:' \
        + args.data + '/feats.scp ark:- |'
feats += ' add-deltas --print-args=false ' + delta_opts + ' ark:- ark:- |'
feats += ' splice-feats --print-args=false --left-context='+str(splice) + \
        ' --right-context='+str(splice) + ' ark:- ark:-|'
feats += ' apply-cmvn --print-args=false --norm-vars=true ' + srcdir+'/cmvn.mat ark:- ark:- |'

count = 0
reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)
vad_reader = kaldi_io.RandomAccessBaseFloatVectorReader('scp:%s/vad.scp' % args.data)
writer = kaldi_io.BaseFloatMatrixWriter(args.wspecifier)

for uid, feats in reader:
  if not vad_reader.has_key(uid):
    raise RuntimeError('No VAD found in %s/vad.scp for utterance %s' % (args.data, uid))

  vad = vad_reader.value(uid)

  embedding = nnet.gen_embedding(feats, vad)

  writer.write(uid, embedding)
  
  count += 1
  if args.verbose and count % 100 == 0:
    logger.info("LOG (nnet_gen_embedding.py) %d utterances processed" % count)

