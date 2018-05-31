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

class SeqDataGenerator:
  def __init__ (self, data, labels, trans_dir, exp, name, conf, 
                seed=777, shuffle=False, num_gpus=1, buckets=None):
    
    self.data = data
    self.labels = labels
    self.exp = exp
    self.name = name
    self.batch_size = conf.get('batch_size', 256) * num_gpus
    self.splice = conf.get('context_width', 5)
    self.sliding_window = conf.get('sliding_window', 200)
    self.max_length = conf.get('max_length', 1000)
    self.feat_type = conf.get('feat_type', 'raw')
    self.clean_up = conf.get('clean_up', True)
    self.variable_length = conf.get('variable_length', False)
    self.buckets = [self.max_length ] if buckets is None else buckets

    # in loop mode, we keep looping over dataset
    # and decide iteration by split_per_iter
    # otherwise, we decide iteration by split_data_counter
    if self.name == 'train':
      self.loop = conf.get('loop_mode', False)
      self.split_per_iter = conf.get('split_per_iter', None)
      self.noise_ratio = conf.get('noise_ratio', 0.0)   # add random noise to training data
    else:
      self.loop = False
      self.split_per_iter = None
      self.noise_ratio = 0.0
    self.split_counter = 0        # keep increasing
    self.split_data_counter = 0   # loop from 0 to num_split
    if self.loop and self.split_per_iter is None:
      raise RuntimeError('Must specify split_per_iter in loop mode!')

    self.tmp_dir = tempfile.mkdtemp(prefix = conf.get('tmp_dir', '/data/suhang/exp/tmp'))

    ## Read number of utterances
    with open (self.data + '/feats.%s.scp' % self.name) as f:
      self.num_utts = sum(1 for line in f)
    
    shutil.copyfile("%s/feats.%s.scp" % (self.data, self.name), "%s/%s.scp" % (self.exp, self.name))

    if name == 'train':
      cmd = ['copy-feats', '\'scp:head -10000 %s/%s.scp |\'' % (self.exp, self.name), 'ark:- |']

      cmd.extend(['splice-feats', '--left-context='+str(self.splice), 
                  '--right-context='+str(self.splice), 'ark:- ark:- |'])

      cmd.extend(['compute-cmvn-stats', 'ark:-', exp+'/cmvn.mat'])

      Popen(' '.join(cmd), shell=True).communicate()

    self.num_split = int(open('%s/num_split.%s' % (self.data, self.name)).read())
  
    for i in range(self.num_split):
      shutil.copyfile("%s/feats.%s.%d.scp" % (self.data, self.name, (i+1)), "%s/split.%s.%d.scp" % (self.tmp_dir, self.name, i))

    self.num_samples = int(open('%s/num_samples.%s' % (self.data, self.name)).read())

    numpy.random.seed(seed)

    self.feat_dim = int(open('%s/feat_dim' % self.data).read()) * (2*self.splice+1)
    
    self.x = numpy.empty ((0, self.max_length, self.feat_dim))
    self.y = numpy.empty (0, dtype='int32')
    self.mask = numpy.empty ((0, self.max_length), dtype='float32')
    
    self.batch_pointer = 0
    
  def get_feat_dim(self):
    return self.feat_dim


  def __del__(self):
    if self.clean_up:
      shutil.rmtree(self.tmp_dir)


  def has_data(self):
    # has enough data for next batch
    if self.batch_pointer + self.batch_size > len(self.x):
      if self.loop and (self.split_counter+1) % self.split_per_iter == 0:
        return False
      if not self.loop and self.split_data_counter == self.num_split:
        return False
    return True
     
      
  def get_num_split(self):
    return self.num_split 


  def get_num_batches(self):
    return self.num_samples / self.batch_size


  ## Return a batch to work on
  def get_next_split_data (self):
    '''
    output: 
      feat_list: list of np matrix [num_frames, feat_dim]
      label_list: list of int32 np array [num_frames] 
    '''
    cmd = [ 'copy-feats', 'scp:' + self.tmp_dir + '/split.' + self.name + '.' + \
           str(self.split_data_counter) + '.scp', 'ark:- |' ]
    cmd.extend(['splice-feats', '--left-context='+str(self.splice),
                 '--right-context='+str(self.splice), 'ark:-', 'ark:-|'])
    cmd.extend(['apply-cmvn', '--norm-vars=true', self.exp+'/cmvn.mat', 'ark:-', 'ark:-'])
    p1 = Popen(' '.join(cmd), shell=True, stdout=PIPE, stderr=DEVNULL)

    feat_list = []
    label_list = []

    while True:
      uid, feat = kaldi_IO.read_utterance (p1.stdout)
      if uid == None:
        break;
      if uid in self.labels:
        feat_list.append (feat)
        label_list.append (self.labels[uid])

    p1.stdout.close()

    if len(feat_list) == 0 or len(label_list) == 0:
      raise RuntimeError("No feats are loaded! please check feature and labels," + \
                         "and make sure they are matched.")

    return (feat_list, label_list)


  def get_bucket(self, length):
    max_length = self.buckets[-1]
    bucket_id = len(self.buckets) - 1
    for i in range(len(self.buckets)):
      if length <= self.buckets[i]:
        max_length = self.buckets[i]
        bucket_id = i
        break
    return max_length, bucket_id


  def pack_utt_data_variable(self, features, sid_labels, bucket_id):
    '''
    for each utterance, we use a rolling window to generate enough segments for speaker ID modeling
    input:
      features: list of np 2d-array [num_frames, feat_dim]
      sid_labels: list of int32
    output:
      features_packed: np 3d-array [batch_size, max_length, feat_dim]
      sid_labels_packed: array[batch_size]
    '''
    assert len(features) == len(sid_labels)

    features_packed = []
    sid_labels_packed = []
    mask = []
    max_length = self.buckets[bucket_id]
    sliding_window = self.sliding_window

    for feat, sid_lab in zip(features, sid_labels):

      start_index = 0
      while start_index + max_length < len(feat):
        # cut utterances into pieces
        end_index = start_index + max_length
        features_packed.append(feat[start_index:end_index])
        mask.append(numpy.ones(max_length))
        sid_labels_packed.append(sid_lab)
        start_index += sliding_window

      # last part, if long enough (bigger than max_length/2), pad the features
      if len(feat) - start_index >= sliding_window:
        num_zero = max_length + start_index - len(feat)
        zeros2pad = numpy.zeros((num_zero, len(feat[0])))
        features_packed.append(numpy.concatenate((feat[start_index:], zeros2pad)))
        mask.append(numpy.append(numpy.ones(len(feat)-start_index), numpy.zeros(num_zero)))
        sid_labels_packed.append(sid_lab)

    features_packed = numpy.array(features_packed)
    mask = numpy.array(mask)
    sid_labels_packed = numpy.array(sid_labels_packed)

    return features_packed, sid_labels_packed, mask


  def pack_utt_data_fixed(self, features, sid_labels):
    '''
    for each utterance, we pad it into predifined length
    input:
      features: list of np 2d-array [num_frames, feat_dim]
    output:
      features_pad: np 3d-array [batch_size, max_length, feat_dim]
      labels_pad: np array [batch_size]
      mask: matrix[batch_size, max_length]
    '''
    features_pad = []
    mask = []
    feat_len = len(features[0])
    max_length, bucket_id = self.get_bucket(feat_len)

    #we assume features from each split are of same length, so we use the same bucket
    for feat in features:
      assert feat_len == len(feat)
      if max_length > len(feat) :
        num_zero = max_length - len(feat)
        zeros2pad = numpy.zeros((num_zero, len(feat[0])))
        features_pad.append(numpy.concatenate((feat, zeros2pad)))
        mask.append(numpy.append(numpy.ones(len(feat)), numpy.zeros(num_zero)))
      else:
        features_pad.append(feat[:max_length])
        mask.append(numpy.ones(max_length))

    features_pad = numpy.array(features_pad)
    sid_labels = numpy.array(sid_labels)
    mask = numpy.array(mask)

    return features_pad, sid_labels, mask, bucket_id


  def get_batch_utterances (self):
    '''
    output:
      x_mini: np matrix [batch_size, max_length, feat_dim]
      y_mini: np matrix [batch_size]
      mask: np matrix [batch_size, max_length]
      bucket_id: an int
    '''
    # read split data until we have enough for this batch
    while (self.batch_pointer + self.batch_size > len(self.x)):
      if not self.has_data():
        return None, None, None, 0

      feats, sid_labels = self.get_next_split_data()
      
      if self.variable_length:
        bucket_id = numpy.random.randint(0, len(self.buckets))
        x_packed, y_packed, mask_packed = self.pack_utt_data_variable(feats, sid_labels, bucket_id)
      else:
        x_packed, y_packed, mask_packed, bucket_id = self.pack_utt_data_fixed(feats, sid_labels)

      self.bucket_id = bucket_id

      ## Shuffle data, utterance base
      randomInd = numpy.array(range(len(x_packed)))
      numpy.random.shuffle(randomInd)
      self.x = x_packed[randomInd]
      self.y = y_packed[randomInd]
      self.mask = mask_packed[randomInd]

      self.batch_pointer = 0

      self.split_counter += 1
      self.split_data_counter += 1
      if self.loop and self.split_data_counter == self.num_split:
        self.split_data_counter = 0
    
    x_mini = self.x[self.batch_pointer:self.batch_pointer+self.batch_size]
    y_mini = self.y[self.batch_pointer:self.batch_pointer+self.batch_size]
    mask_mini = self.mask[self.batch_pointer:self.batch_pointer+self.batch_size]

    if self.noise_ratio != 0.0:
      random_noise = numpy.random.normal(0.0, self.noise_ratio, x_mini.shape)
      x_mini += random_noise

    self.last_batch_utts = len(y_mini)
    self.batch_pointer += self.batch_size

    return x_mini, y_mini, mask_mini, self.bucket_id


  def get_batch_size(self):
    return self.batch_size

  
  def get_last_batch_counts(self):
    return self.last_batch_utts


  def count_units(self):
    return 'utts'


  def reset_batch(self):
    if self.loop:
      self.split_counter += 1
    else:
      self.split_data_counter = 0


