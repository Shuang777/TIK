from subprocess import Popen, PIPE, check_output
import tempfile
import kaldi_io
import kaldi_IO
import pickle
import shutil
import numpy
import os
import math

DEVNULL = open(os.devnull, 'w')

class UttDataGenerator:
  def __init__ (self, data, labels, trans_dir, exp, name, conf, 
                seed=777, shuffle=False, num_gpus = 1):
    
    self.data = data
    self.labels = labels
    self.exp = exp
    self.name = name
    self.batch_size = conf.get('batch_size', 128) * num_gpus
    self.splice = conf.get('context_width', 1)
    self.max_length = conf.get('max_length', 2000)
    self.sliding_window = conf.get('sliding_window', 20)
    self.jitter_window = conf.get('jitter_window', 0)
    self.feat_type = conf.get('feat_type', 'raw')
    self.delta_opts = conf.get('delta_opts', '')
    self.max_split_data_size = conf.get('max_split_data_size', 2000)
    self.clean_up = conf.get('clean_up', True)

    if self.name == 'train':
      self.loop = conf.get('loop_mode', False)
      self.split_per_iter = conf.get('split_per_iter', None)
    else:
      self.loop = False
      self.split_per_iter = None
    self.split_counter = 0        # keep increasing
    self.split_data_counter = 0   # loop from 0 to num_split
    if self.loop and self.split_per_iter is None:
      raise RuntimeError('Must specify split_per_iter in loop mode!')

    self.tmp_dir = tempfile.mkdtemp(prefix = conf.get('tmp_dir', '/data/suhang/exp/tmp'))

    ## Read number of utterances
    with open (data + '/feats.%s.scp' % self.name) as f:
      self.num_utts = sum(1 for line in f)

    shutil.copyfile("%s/feats.%s.scp" % (self.data, self.name), "%s/%s.scp" % (self.exp, self.name))

    if name == 'train':
      cmd = ['copy-feats', '\'scp:head -10000 %s/%s.scp |\'' % (exp, self.name), 'ark:- |']
      cmd.extend(['splice-feats', '--left-context='+str(self.splice),
                  '--right-context='+str(self.splice), 'ark:- ark:- |'])
      cmd.extend(['compute-cmvn-stats', 'ark:-', exp+'/cmvn.mat'])
      Popen(' '.join(cmd), shell=True).communicate()

    self.num_split = int(math.ceil(1.0 * self.num_utts / self.max_split_data_size))
    for i in range(self.num_split):
      shutil.copyfile("%s/feats.%s.%d.scp" % (self.data, self.name, (i+1)), "%s/split.%s.%d.scp" % (self.tmp_dir, self.name, i))
 
    numpy.random.seed(seed)

    self.feat_dim = int(open('%s/feat_dim' % self.data).read()) * (2*self.splice+1)

    self.x = numpy.empty ((0, self.max_length, self.feat_dim))
    self.y = numpy.empty ((0, self.max_length), dtype='int32')
    self.seq_length = numpy.empty (0, dtype='int32')
    self.mask = numpy.empty ((0, self.max_length), dtype='float32')
    
    self.batch_pointer = 0


  def get_feat_dim(self):
    return self.feat_dim


  def __del__(self):
    if self.clean_up and os.path.exists(self.feat_dir):
      shutil.rmtree(self.tmp_dir)


  def has_data(self):
    # has enough data for next batch
    if self.batch_pointer + self.batch_size > len(self.x):
      if self.loop and (self.split_counter+1) % self.split_per_iter == 0:
        return False
      if not self.loop and self.split_data_counter == self.num_split:
        return False
    return True
    

  ## Return a batch to work on
  def get_next_split_data (self):
    '''
    output: 
      feat_list: list of np matrix [num_frames, feat_dim]
      label_list: list of int32 np array [num_frames] 
    '''

    p1 = Popen (['splice-feats', '--print-args=false', '--left-context='+str(self.splice),
                 '--right-context='+str(self.splice), 
                 'scp:'+self.tmp_dir+'/split.'+self.name+'.'+str(self.split_data_counter)+'.scp',
                 'ark:-'], stdout=PIPE, stderr=DEVNULL)
    p2 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', self.exp+'/cmvn.mat',
                 'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE, stderr=DEVNULL)

    feat_list = []
    label_list = []
    
    while True:
      uid, feat = kaldi_IO.read_utterance (p2.stdout)
      if uid == None:
        break;
      if uid in self.labels:
        feat_list.append (feat)
        label_list.append (self.labels[uid])

    p1.stdout.close()
    
    if len(feat_list) == 0 or len(label_list) == 0:
      raise RuntimeError("No feats are loaded! please check feature and labels, and make sure they are matched.")

    return (feat_list, label_list)

          
  def pack_utt_data(self, features, labels):
    '''
    for each utterance, we use a rolling window to predict the output posterior. 
    Reference: Deep Bi-Directional Recurrent Network over Spectral Windows
    input:
      features: list of np 2d-array [num_frames, feat_dim]
      labels: list of np array [num_frames]
    output:
      features_pad: np 3d-array [batch_size, max_length, feat_dim]
      labels_pad: matrix[batch_size, max_length]
      seq_length: np array [batch_size]
      mask: matrix[batch_size, max_length]
    '''
    assert len(features) == len(labels)

    features_pad = []
    labels_pad = []
    mask = []
    seq_length = []
    max_length = self.max_length
    sliding_window = self.sliding_window
    jitter_window = self.jitter_window

    for feat, lab in zip(features, labels):

      assert len(feat) == len(lab)
      start_index = 0

      pick_start = 0
      pick_end = (max_length + jitter_window) // 2
      while start_index + max_length < len(feat):
        # cut utterances into pieces
        end_index = start_index + max_length
        features_pad.append(feat[start_index:end_index])
        labels_pad.append(lab[start_index:end_index])
        seq_length.append(max_length)
        if jitter_window != 0:
          this_mask = numpy.zeros(max_length)
          this_mask[pick_start:pick_end] = 1
          mask.append(this_mask)
          # only the first window starts from 0, all others start from (max_length - jitter_window) / 2
          pick_start = (max_length - jitter_window) // 2
          start_index += sliding_window
        else:
          mask.append(numpy.ones(max_length))
          start_index += sliding_window

      # last part, pad zero
      num_zero = max_length + start_index - len(feat)
      zeros2pad = numpy.zeros((num_zero, len(feat[0])))
      features_pad.append(numpy.concatenate((feat[start_index:], zeros2pad)))
      labels_pad.append(numpy.append(lab[start_index:], numpy.zeros(num_zero)))
      seq_length.append(len(feat) - start_index)
      if jitter_window != 0:
        # our last window goes till the end of the utterance
        this_mask = numpy.zeros(max_length)
        this_mask[pick_start:(len(feat) - start_index)] = 1
        mask.append(this_mask)
      else:
        mask.append(numpy.append(numpy.ones(len(feat) - start_index), numpy.zeros(num_zero)))

    features_pad = numpy.array(features_pad)
    labels_pad = numpy.array(labels_pad)
    seq_length = numpy.array(seq_length)
    mask = numpy.array(mask)

    return features_pad, labels_pad, seq_length, mask


  def get_batch_utterances (self):
    '''
    output:
      x_mini: np matrix [batch_size, max_length, feat_dim]
      y_mini: np matrix [batch_size, max_length]
      seq_length: np array [batch_size]
      mask: np matrix [batch_size, max_length]
    '''
    # read split data until we have enough for this batch
    while (self.batch_pointer + self.batch_size > len (self.x)):
      if not self.has_data():
        # not loop mode and we arrive the end, do not read anymore
        return None, None, None, None

      x, y = self.get_next_split_data()
      x_pad, y_pad, seq_length, mask = self.pack_utt_data(x, y)

      self.x = numpy.concatenate ((self.x[self.batch_pointer:], x_pad))
      self.y = numpy.concatenate ((self.y[self.batch_pointer:], y_pad))
      self.seq_length = numpy.append(self.seq_length[self.batch_pointer:], seq_length)
      self.mask = numpy.concatenate ((self.mask[self.batch_pointer:], mask))
      
      self.batch_pointer = 0

      ## Shuffle data, utterance base
      randomInd = numpy.array(range(len(self.x)))
      numpy.random.shuffle(randomInd)
      self.x = self.x[randomInd]
      self.y = self.y[randomInd]
      self.seq_length = self.seq_length[randomInd]
      self.mask = self.mask[randomInd]

      self.split_counter += 1
      self.split_data_counter += 1
      if self.loop and self.split_data_counter == self.num_split:
        self.split_data_counter = 0
    
    x_mini = self.x[self.batch_pointer:self.batch_pointer+self.batch_size]
    y_mini = self.y[self.batch_pointer:self.batch_pointer+self.batch_size]
    seq_mini = self.seq_length[self.batch_pointer:self.batch_pointer+self.batch_size]
    mask_mini = self.mask[self.batch_pointer:self.batch_pointer+self.batch_size]

    self.batch_pointer += self.batch_size
    self.last_batch_frames = mask_mini.sum()

    return x_mini, y_mini, seq_mini, mask_mini


  def get_batch_size(self):
    return self.batch_size


  def get_last_batch_counts(self):
    return self.last_batch_frames

  
  def count_units(self):
    return 'frames'


  def reset_batch(self):
    if self.loop:
      self.split_counter += 1
    else:
      self.split_data_counter = 0


  def save_target_counts(self, num_targets, output_file):
    # here I'm assuming training data is less than 10,000 hours
    counts = numpy.zeros(num_targets, dtype='int64')
    for alignment in self.labels.values():
      counts += numpy.bincount(alignment, minlength = num_targets)
    # add a ``half-frame'' to all the elements to avoid zero-counts (decoding issue)
    counts = counts.astype(float) + 0.5
    numpy.savetxt(output_file, counts, fmt = '%.1f')


