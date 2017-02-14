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
os.path.isdir(exp+'/nnet') or os.makedirs (exp+'/nnet')
shutil.copyfile(gmm+'/tree', exp+'/tree')
shutil.copyfile(gmm+'/final.mdl', exp+'/final.mdl')
shutil.copyfile(gmm+'/final.mat', exp+'/final.mat')

# get the feature input dim
output_dim = readOutputFeatDim(gmm)

# prepare data
trGen = dataGenerator (data_tr, ali_tr, ali_tr, exp, 'train', config.items('nnet'), shuffle=True)
cvGen = dataGenerator (data_cv, ali_cv, ali_cv, exp, 'cv', config.items('nnet'))
#cvGen = None

# create the neural net
nnet = Nnet(config.items('nnet'), trGen.getFeatDim(), output_dim)
nnet.init_nnet()
mlp_init = exp+'/model.init'

if os.path.isfile(exp+'/.mlp_best'):
    mlp_best = open(exp+'/.mlp_best').read()
    nnet.read(mlp_best)
elif os.path.isfile(mlp_init):
    nnet.read(mlp_init)
    mlp_best = mlp_init
else:
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

loss = nnet.test(cvGen)

current_lr = initial_lr
if os.path.isfile(exp+'/.learn_rate'):
    current_lr = float(open(exp+'/.learn_rate').read())
if os.path.isfile(exp+'/.halving'):
    halving = bool(open(exp+'/.halving').read())
else:
    halving = False

print '### neural net training started at ' + str(datetime.datetime.today())
for i in xrange(max_iters):
    print "ITERATION %d: " % i,

    if os.path.isfile(exp+'/.done_iter'+str(i+1)):
        print "skipping... "
        continue

    loss_tr = nnet.train(trGen, current_lr)
    loss_new = nnet.test(cvGen)
    loss_prev = loss

    mlp_current = "%s/nnet/model_iter%d_lr%f_tr%.3f_cv%.3f" % (exp, i+1, current_lr, loss_tr, loss_new)

    if loss_new < loss or i < keep_lr_iters or i < min_iters:
        # accepting: the loss was better or we have fixed learn-rate
        loss = loss_new
        mlp_best = mlp_current
        nnet.write(mlp_best)
        print "nnet accepted %s" % mlp_best.split('/')[-1]
        open(exp + '/.mlp_best', 'w').write(mlp_best)
    else:
        mlp_rej = mlp_current + "_rejected"
        nnet.write(mlp_rej)
        print "nnet rejected %s" % mlp_rej.split('/')[-1]

    open(exp + '/.done_iter.'+str(i+1), 'w').write("")
    
    if i < keep_lr_iters:
        continue
    
    # stopping criterion
    rel_impr = (loss_prev - loss) / loss_prev
    if halving and rel_impr < end_halving_impr:
        if i < min_iters:
            print "we were supposed to finish, but we continue as min_iters: %d" % min_iters
            continue
        print "finished, too small rel. improvement %.3f" % rel_impr
        break

    if rel_impr < start_halving_impr:
        halving = True
        open(exp+'/.halving', 'w').write(str(halving))

    if halving:
        current_lr = current_lr * halving_factor
        open(exp+'/.learn_rate', 'w').write(current_lr)

# end of train loop

print "Succeed training the neural network in " + exp + '/final.model'
print '### training complete at ' + str(datetime.datetime.today())
