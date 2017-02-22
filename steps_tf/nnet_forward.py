#!/usr/bin/python3

import os
import sys
import numpy
import logging
import kaldiIO
import argparse
from time import sleep
from subprocess import Popen, PIPE, DEVNULL
from six.moves import configparser
from signal import signal, SIGPIPE, SIG_DFL
from nnet_trainer import Nnet

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

config_file = args.config_file
model_file  = args.model_file
prior_counts_file  = args.prior_counts_file

config = configparser.ConfigParser()
config.read(config_file)
srcdir = os.path.dirname(model_file)

input_dim = int(open(srcdir+'/input_dim').read())
output_dim = int(open(srcdir+'/output_dim').read())
splice = config.getint('nnet','context_width')

# set gpu
logger.info("use-gpu: %s", str(args.use_gpu))

if args.use_gpu in [ 'yes', 'true', 'True']:
  #sleep(args.sleep)
  p1 = Popen ('pick-gpu', stdout=PIPE)
  gpu_id = int(p1.stdout.read())
  if gpu_id == -1:
    raise RuntimeError("Unable to pick gpu")
  logger.info("Selecting gpu %d", gpu_id)
  os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = ''

nnet = Nnet(config.items('nnet'), config.items('optimizer'), input_dim, output_dim)

nnet.read(open(model_file, 'r').read())

prior_counts = numpy.genfromtxt (prior_counts_file)
priors = prior_counts / prior_counts.sum()

ark_in = sys.stdin.buffer
#ark_in = open('stdin', 'r')
ark_out = sys.stdout.buffer
encoding = sys.stdout.encoding
signal (SIGPIPE, SIG_DFL)

p1 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', srcdir+'/cmvn.mat', 
             'ark:-', 'ark:-'], stdin=ark_in, stdout=PIPE, stderr=DEVNULL)
p2 = Popen(['splice-feats', '--print-args=false', '--left-context='+str(splice), 
            '--right-context='+str(splice), 'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE)

while True:
  uid, feats = kaldiIO.readUtterance(p2.stdout)
  if uid == None:
    # we are done
    break

  posteriors = nnet.predict (feats)
  logProbMat = numpy.log(posteriors / priors)
  kaldiIO.writeUtterance(uid, logProbMat, ark_out, encoding)

p1.stdout.close
