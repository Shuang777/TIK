import os
import sys
import shutil
import datetime
from six.moves import configparser
from subprocess import Popen, PIPE
from nnet_trainer import Nnet
from dataGenerator import dataGenerator


#get number of output labels
def readOutputFeatDim (gmm):
    p1 = Popen (['am-info', '%s/final.mdl' % gmm], stdout=PIPE)
    modelInfo = p1.stdout.read().splitlines()
    for line in modelInfo:
        if b'number of pdfs' in line:
            return int(line.split()[-1])


if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 7:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir')

print "### command line"
print " ".join(sys.argv)

data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
exp     = sys.argv[6]

# read config file
config = configparser.ConfigParser()
config.read('config/swbd.cfg')

# prepare data dir
os.path.isdir(exp) or os.makedirs (exp)
shutil.copyfile(gmm+'/tree', exp+'/tree')
shutil.copyfile(gmm+'/final.mdl', exp+'/final.mdl')
shutil.copyfile(gmm+'/final.mat', exp+'/final.mat')

# get the feature input dim
output_dim = readOutputFeatDim(gmm)

# prepare data
trGen = dataGenerator (data_tr, ali_tr, ali_tr, exp, 'train', config.getint('nnet', 'batch_size'), shuffle=True)
cvGen = dataGenerator (data_cv, ali_cv, ali_cv, exp, 'cv', config.getint('nnet', 'batch_size'))
#cvGen = None

# create the neural net
nnet = Nnet(config, trGen.getFeatDim(), output_dim)
nnet.init_nnet()

#train the neural net
print '### neural net training started at ' + str(datetime.datetime.today())
nnet.train(trGen)
print '### neural net training finished at ' + str(datetime.datetime.today())

print '### neural net testing started at ' + str(datetime.datetime.today())
nnet.test(cvGen)

nnet.write(exp + '/model.chpt')

print '### training complete at ' + str(datetime.datetime.today())
