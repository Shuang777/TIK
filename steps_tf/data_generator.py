from subprocess import Popen, PIPE, DEVNULL, check_output
import tempfile
import kaldi_IO
import pickle
import shutil
import numpy
import os

## Data generator class for Kaldi
class DataGenerator:
  def __init__ (self, data, labels, 
                trans_dir, exp, name, 
                conf, 
                seed=777, 
                shuffle=False,
                loop=False):
    
    self.data = data
    self.labels = labels
    self.exp = exp
    self.name = name
    self.batch_size = conf.get('batch_size', 256)
    self.splice = conf.get('context_width', 5)
    self.loop = loop    # keep looping over dataset
    self.max_split_data_size = 1000 ## These many utterances are loaded into memory at once.

    self.temp_dir = tempfile.mkdtemp(prefix='/data/exp/tmp/')

    ## Read number of utterances
    with open (data + '/utt2spk') as f:
      self.num_utts = sum(1 for line in f)

    cmd = "cat %s/feats.scp | utils/shuffle_list.pl --srand %d > %s/shuffle.%s.scp" % (data, seed, exp, self.name)
    Popen(cmd, shell=True).communicate()

    cmd = ['apply-cmvn', '--utt2spk=ark:' + self.data + '/utt2spk',
                 'scp:' + self.data + '/cmvn.scp',
                 'scp:' + exp + '/shuffle.' + self.name + '.scp','ark:-']
    
    cmd += ['|', 'splice-feats', 'ark:-','ark:-', '|', 'transform-feats', exp+'/final.mat', 'ark:-', 'ark:-']

    if os.path.exists(trans_dir+'/trans.1'):
      cmd += ['|', 'transform-feats','--utt2spk=ark:' + self.data + '/utt2spk',
              '\'ark:cat %s/trans.* |\'' % trans_dir, 'ark:-', 'ark:-']
    
    cmd += ['|', 'copy-feats', 'ark:-', 'ark,scp:'+self.temp_dir+'/shuffle.'+self.name+'.ark,'+exp+'/'+self.name+'.scp']
    Popen(' '.join(cmd), shell=True).communicate()

    if name == 'train':
      p1 = Popen (['splice-feats', '--left-context='+str(self.splice), '--right-context='+str(self.splice),
                   'scp:head -10000 %s/%s.scp |' % (exp, self.name), 'ark:-'], 
                   stdout=PIPE)
      Popen(['compute-cmvn-stats', 'ark:-', exp+'/cmvn.mat'], stdin=p1.stdout).communicate()
      p1.stdout.close()

    split_scp_cmd = 'utils/split_scp.pl ' + exp + '/' + self.name + '.scp'
    self.num_split = - (-self.num_utts // self.max_split_data_size)  # integer division
    for i in range(self.num_split):
      split_scp_cmd += ' ' + self.temp_dir + '/split.' + self.name + '.' + str(i) + '.scp'
    
    Popen (split_scp_cmd, shell=True).communicate()
    
    numpy.random.seed(seed)

    self.feat_dim = int(check_output(['feat-to-dim', 'scp:%s/%s.scp' %(exp, self.name), '-'])) * (2*self.splice+1)
    self.split_data_counter = 0
    
    self.x = numpy.empty ((0, self.feat_dim))
    self.y = numpy.empty (0, dtype='uint32')
    self.batch_pointer = 0


  def get_feat_dim(self):
    return self.feat_dim


  def __exit__ (self):
    shutil.rmtree(self.temp_dir)
  

  ## Return a batch to work on
  def get_next_split_data (self):
    p1 = Popen (['splice-feats', '--print-args=false', '--left-context='+str(self.splice), 
                 '--right-context='+str(self.splice), 
                 'scp:'+self.temp_dir+'/split.'+self.name+'.'+str(self.split_data_counter)+'.scp',
                 'ark:-'], stdout=PIPE, stderr=DEVNULL)
    p2 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', self.exp+'/cmvn.mat', 
                 'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE, stderr=DEVNULL)

    feat_list = []
    label_list = []
    while True:
      uid, feat = kaldi_IO.read_utterance (p2.stdout)
      if uid == None:
        # no more utterance, return
        return (numpy.vstack(feat_list), numpy.hstack(label_list))
      if uid in self.labels:
        feat_list.append (feat)
        label_list.append (self.labels[uid])
    # read done

    p1.stdout.close()


  def hasData(self):
  # has enough data for next batch
    if self.loop or self.split_data_counter != self.num_split:     # we always have data if in loop mode
      return True
    elif self.batch_pointer + self.batch_size >= len(self.x):
      return False
    return True
      
          
  ## Retrive a mini batch
  def get_batch (self):
    # read split data until we have enough for this batch
    while (self.batch_pointer + self.batch_size >= len (self.x)):
      if not self.loop and self.split_data_counter == self.num_split:
        # not loop mode and we arrive the end, do not read anymore
        return None, None

      x,y = self.get_next_split_data()
      self.x = numpy.concatenate ((self.x[self.batch_pointer:], x))
      self.y = numpy.append (self.y[self.batch_pointer:], y)
      self.batch_pointer = 0

      ## Shuffle data
      randomInd = numpy.array(range(len(self.x)))
      numpy.random.shuffle(randomInd)
      self.x = self.x[randomInd]
      self.y = self.y[randomInd]

      self.split_data_counter += 1
      if self.loop and self.split_data_counter == self.num_split:
        self.split_data_counter = 0
    
    x_mini = self.x[self.batch_pointer:self.batch_pointer+self.batch_size]
    y_mini = self.y[self.batch_pointer:self.batch_pointer+self.batch_size]
    self.batch_pointer += self.batch_size
    return x_mini, y_mini


  def get_batch_size(self):
      return self.batch_size


  def reset_batch(self):
      self.split_data_counter = 0


  def save_target_counts(self, num_targets, output_file):
      # here I'm assuming training data is less than 10,000 hours
      counts = numpy.zeros(num_targets, dtype='int64')
      for alignment in self.labels.values():
        counts += numpy.bincount(alignment, minlength = num_targets)
      # add a ``half-frame'' to all the elements to avoid zero-counts (decoding issue)
      counts = counts.astype(float) + 0.5
      numpy.savetxt(output_file, counts, fmt = '%.1f')

