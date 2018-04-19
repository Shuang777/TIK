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

class JointDNNDataGenerator:
  def __init__ (self, data, sid_labels, asr_labels, exp, name, conf, 
                seed=777, shuffle=False, loop=False, num_gpus = 1, buckets=None):
    
    self.data = data
    self.sid_labels = sid_labels
    self.asr_labels = asr_labels
    self.exp = exp
    self.name = name
    self.batch_size = conf.get('batch_size', 256) * num_gpus
    self.splice = conf.get('context_width', 5)
    self.sliding_window = conf.get('sliding_window', 200)
    self.max_length = conf.get('max_length', 400)
    self.feat_type = conf.get('feat_type', 'raw')
    self.delta_opts = conf.get('delta_opts', '')
    self.buckets = buckets
    self.loop = loop    # keep looping over dataset
    
    if self.buckets is None:    # we only have one bucket in this case
      self.buckets = [self.max_length]

    self.max_split_data_size = 2000 ## These many utterances are loaded into memory at once.

    self.tmp_dir = tempfile.mkdtemp(prefix = conf.get('tmp_dir', '/data/exp/tmp'))

    ## Read number of utterances
    with open (data + '/utt2spk') as f:
      self.num_utts = sum(1 for line in f)

    cmd = "cat %s/feats.scp | utils/shuffle_list.pl --srand %d > %s/shuffle.%s.scp" % (data, seed, exp, self.name)
    Popen(cmd, shell=True).communicate()

    # prepare feature pipeline
    if conf.get('cmvn_type', 'utt') == 'utt':
      cmd = ['apply-cmvn', '--utt2spk=ark:' + self.data + '/utt2spk',
                 'scp:' + self.data + '/cmvn.scp',
                 'scp:' + exp + '/shuffle.' + self.name + '.scp','ark:- |']
    elif conf['cmvn_type'] == 'sliding':
      cmd = ['apply-cmvn-sliding', '--norm-vars=false', '--center=true', '--cmn-window=300', 
              'scp:' + exp + '/shuffle.' + self.name + '.scp','ark:- |']
    else:
      raise RuntimeError("cmvn_type %s not supported" % conf['cmvn_type'])

    if self.feat_type == 'delta':
      feat_dim_delta_multiple = 3
    else:
      feat_dim_delta_multiple = 1
    
    if self.feat_type == 'delta':
      cmd.extend(['add-deltas', self.delta_opts, 'ark:-', 'ark:- |'])
    elif self.feat_type in ['lda', 'fmllr']:
      cmd.extend(['splice-feats', 'ark:-','ark:- |'])
      cmd.extend(['transform-feats', exp+'/final.mat', 'ark:-', 'ark:- |'])

    if self.feat_type == 'fmllr':
      assert os.path.exists(trans_dir+'/trans.1') 
      cmd.extend(['transform-feats','--utt2spk=ark:' + self.data + '/utt2spk',
              '\'ark:cat %s/trans.* |\'' % trans_dir, 'ark:-', 'ark:-|'])
    
    cmd.extend(['copy-feats', 'ark:-', 'ark,scp:'+self.tmp_dir+'/shuffle.'+self.name+'.ark,'+exp+'/'+self.name+'.scp'])
    Popen(' '.join(cmd), shell=True).communicate()

    if name == 'train':
      cmd =['splice-feats', '--left-context='+str(self.splice), '--right-context='+str(self.splice),
            '\'scp:head -10000 %s/%s.scp |\'' % (exp, self.name), 'ark:- |', 'compute-cmvn-stats', 
            'ark:-', exp+'/cmvn.mat']
      Popen(' '.join(cmd), shell=True).communicate()

    self.num_split = int(math.ceil(1.0 * self.num_utts / self.max_split_data_size))
    for i in range(self.num_split):
      split_scp_cmd = 'utils/split_scp.pl -j %d ' % (self.num_split)
      split_scp_cmd += '%d %s/%s.scp %s/split.%s.%d.scp' % (i, exp, self.name, self.tmp_dir, self.name, i)
      Popen (split_scp_cmd, shell=True).communicate()
    
    numpy.random.seed(seed)

    self.feat_dim = int(check_output(['feat-to-dim', 'scp:%s/%s.scp' %(exp, self.name), '-'])) * \
                    feat_dim_delta_multiple * (2*self.splice+1)
    self.split_data_counter = 0
    
    self.xs = []
    self.ys = []
    self.zs = []
    self.masks = []
    self.batch_pointers = []

    for length in self.buckets:
      x = numpy.empty ((0, length, self.feat_dim))  # features
      y = numpy.empty ((0, length), dtype='int32')  # asr_labels
      z = numpy.empty (0, dtype='int32')                     # sid_labels
      mask = numpy.empty ((0, length), dtype='float32')

      self.xs.append(x)
      self.ys.append(y)
      self.zs.append(z)
      self.masks.append(mask)
    
      self.batch_pointers.append(0)

    self.bucket_queue = numpy.empty(0, dtype='int32')
    self.bucket_queue_pointer = 0

    
  def get_feat_dim(self):
    return self.feat_dim


  def clean (self):
    shutil.rmtree(self.tmp_dir)


  def has_data(self):
  # has enough data for next batch
    if self.loop or self.split_data_counter != self.num_split:     
      # we always have data if in loop mode; or we haven't reach num_split
      return True
    elif self.bucket_queue_pointer >= len(self.bucket_queue):
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
      asr_label_list: list of int32 np array [num_frames] 
      sid_label_list: list of int32
    '''
    p1 = Popen (['splice-feats', '--print-args=false', '--left-context='+str(self.splice),
                 '--right-context='+str(self.splice), 
                 'scp:'+self.tmp_dir+'/split.'+self.name+'.'+str(self.split_data_counter)+'.scp',
                 'ark:-'], stdout=PIPE, stderr=DEVNULL)
    p2 = Popen (['apply-cmvn', '--print-args=false', '--norm-vars=true', self.exp+'/cmvn.mat',
                 'ark:-', 'ark:-'], stdin=p1.stdout, stdout=PIPE, stderr=DEVNULL)

    feat_list = []
    asr_label_list = []
    sid_label_list = []
    
    while True:
      uid, feat = kaldi_IO.read_utterance (p2.stdout)
      if uid == None:
        break;
      if uid in self.asr_labels and uid in self.sid_labels:
        feat_list.append (feat)
        asr_label_list.append (self.asr_labels[uid])
        sid_label_list.append (self.sid_labels[uid])

    p2.stdout.close()
    
    if len(feat_list) == 0 or len(asr_label_list) == 0:
      raise RuntimeError("No feats are loaded! please check feature and labels, and make sure they are matched.")

    return (feat_list, asr_label_list, sid_label_list)


  def pack_utt_data(self, features, asr_labels, sid_labels, bucket_id):
    '''
    for each utterance, we use a rolling window to generate enough segments for speaker ID modeling
    input:
      features: list of np 2d-array [num_frames, feat_dim]
      asr_labels: list of np array [num_frames]
      sid_labels: list of int32
    output:
      features_packed: np 3d-array [batch_size, max_length, feat_dim]
      asr_labels_packed: matrix[batch_size, max_length]
      sid_labels_packed: array[batch_size]
    '''
    assert len(features) == len(asr_labels)

    features_packed = []
    asr_labels_packed = []
    sid_labels_packed = []
    mask = []
    max_length = self.buckets[bucket_id]
    sliding_window = self.sliding_window

    for feat, asr_lab, sid_lab in zip(features, asr_labels, sid_labels):

      assert len(feat) == len(asr_lab)
      start_index = 0

      if bucket_id != 0 and len(feat) < max_length:
        # if len(feat) < max_length, we don't add it into this bucket, 
        # only if the smallest bucket is still bigger than len(feat)
        # i.e. we have buckets of size 3, 5, 10
        # then frames of 1,2,3,4,5,6 ... goes to bucket 3
        # frames of 5,6,7,... goes to bucket 5
        # frames of 10,11,12,... goes to bucket 10
        continue

      while start_index + max_length < len(feat):
        # cut utterances into pieces
        end_index = start_index + max_length
        features_packed.append(feat[start_index:end_index])
        asr_labels_packed.append(asr_lab[start_index:end_index])
        mask.append(numpy.ones(max_length))
        sid_labels_packed.append(sid_lab)
        start_index += sliding_window

      # last part, if long enough (bigger than max_length/2), pad the features
      if len(feat) - start_index > max_length / 2:
        num_zero = max_length + start_index - len(feat)
        zeros2pad = numpy.zeros((num_zero, len(feat[0])))
        features_packed.append(numpy.concatenate((feat[start_index:], zeros2pad)))
        asr_labels_packed.append(numpy.append(asr_lab[start_index:], 
                                              numpy.zeros(num_zero,dtype='int32')))
        mask.append(numpy.append(numpy.ones(len(feat)-start_index), numpy.zeros(num_zero)))
        sid_labels_packed.append(sid_lab)

    num_batch2add = len(sid_labels_packed)
    features_packed = numpy.array(features_packed)
    asr_labels_packed = numpy.array(asr_labels_packed)
    sid_labels_packed = numpy.array(sid_labels_packed)

    return features_packed, asr_labels_packed, sid_labels_packed, mask, num_batch2add


  def get_batch_utterances (self):
    '''
    output:
      x_mini: np matrix [batch_size, max_length, feat_dim]
      y_mini: np matrix [batch_size]
      mask: np matrix [batch_size, max_length]
    '''
    # read split data until we have enough for this batch
    while self.bucket_queue_pointer >= len(self.bucket_queue):
      if not self.loop and self.split_data_counter == self.num_split:
        # not loop mode and we arrive the end, do not read anymore
        # let's just throw away the last few samples
        return None, None, None, None, None

      feats, asr_labels, sid_labels = self.get_next_split_data()

      buckets2extend = []
      for bucket_id in xrange(len(self.buckets)):
        x_packed, y_packed, z_packed, \
          mask_packed, num_segments2add = self.pack_utt_data(feats, asr_labels, 
                                                          sid_labels, bucket_id)
        x = self.xs[bucket_id]
        y = self.ys[bucket_id]
        z = self.zs[bucket_id]
        mask = self.masks[bucket_id]

        x = numpy.concatenate((x[self.batch_pointers[bucket_id]:], x_packed))
        y = numpy.concatenate((y[self.batch_pointers[bucket_id]:], y_packed))
        z = numpy.append(z[self.batch_pointers[bucket_id]:], z_packed)
        mask = numpy.concatenate((mask[self.batch_pointers[bucket_id]:], mask_packed))
        
        ## Shuffle data, utterance base
        # data is already shuffled. we don't need to do that again
        randomInd = numpy.array(range(len(x)))
        numpy.random.shuffle(randomInd)
        self.xs[bucket_id] = x[randomInd]
        self.ys[bucket_id] = y[randomInd]
        self.zs[bucket_id] = z[randomInd]
        self.masks[bucket_id] = mask[randomInd]
      
        self.batch_pointers[bucket_id] = 0
        num_batches2add = num_segments2add / self.batch_size
        buckets2extend.extend([bucket_id]*num_batches2add)

      numpy.random.shuffle(buckets2extend)
      self.bucket_queue = numpy.append(self.bucket_queue[self.bucket_queue_pointer:], 
                                       buckets2extend)
      self.bucket_queue_pointer = 0

      self.split_data_counter += 1
      if self.loop and self.split_data_counter == self.num_split:
        self.split_data_counter = 0
    
    this_bucket = self.bucket_queue[self.bucket_queue_pointer]
    this_batch_pointer = self.batch_pointers[this_bucket]

    x_mini = self.xs[this_bucket][this_batch_pointer:this_batch_pointer+self.batch_size]
    y_mini = self.ys[this_bucket][this_batch_pointer:this_batch_pointer+self.batch_size]
    z_mini = self.zs[this_bucket][this_batch_pointer:this_batch_pointer+self.batch_size]
    mask_mini = self.masks[this_bucket][this_batch_pointer:this_batch_pointer+self.batch_size]

    self.batch_pointers[this_bucket] += self.batch_size
    self.bucket_queue_pointer += 1

    self.last_batch_utts = len(y_mini)
    self.last_batch_frames = mask_mini.sum()

    return x_mini, y_mini, z_mini, mask_mini, this_bucket


  def get_batch_size(self):
    return self.batch_size

  def get_last_batch_utts(self):
    return self.last_batch_utts

  def get_last_batch_frames(self):
    return self.last_batch_frames

  def reset_batch(self):
    self.split_data_counter = 0

  def count_units(self):
    return 'utterances'
