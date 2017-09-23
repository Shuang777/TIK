import shutil
import os
import time
import numpy as np
import tensorflow as tf
from subprocess import Popen,PIPE
import nnet
import math
import logging
from make_nnet_proto import make_nnet_proto, make_lstm_proto, make_seq2class_proto
from dnn import DNN
from bn import BN
from lstm import LSTM
from seq2class import SEQ2CLASS

logger = logging.getLogger('__main__')
logger.setLevel(logging.INFO)

iter_logger = logging.getLogger(__name__)
iter_logger.setLevel(logging.INFO)

class NNTrainer(object):
  '''
  a class for a neural network that can be used together with Kaldi.
  session is initialized either by read() or by init_nnet().
  '''

  def __init__(self, arch, input_dim, output_dim, batch_size, num_gpus = 1, use_gpu = True,
               summary_dir = None, max_length = None, jitter_window = None):
    ''' just some basic config for this trainer '''
    self.arch = arch
    self.graph = None
    self.sess = None
    self.batch_size = batch_size
    self.max_length = max_length
    self.jitter_window = jitter_window
    self.num_gpus = num_gpus
    self.use_gpu = use_gpu
    self.summary_dir = summary_dir
    self.wait_gpu = True

    if self.arch == 'dnn':
      self.model = DNN(input_dim, output_dim, batch_size, num_gpus)
    elif self.arch == 'bn':
      self.model = BN(input_dim, output_dim, batch_size, num_gpus)
    elif self.arch == 'lstm':
      self.model = LSTM(input_dim, output_dim, batch_size, max_length, num_gpus)
    elif self.arch == 'seq2class':
      self.model = SEQ2CLASS(input_dim, output_dim, batch_size, max_length, num_gpus)
    else:
      raise RuntimeError("arch type %s not supported", self.arch)
    

  def make_proto(self, nnet_conf, nnet_proto_file):
    if self.arch in ['dnn', 'bn']:    # currently we use the same function for dnn and bn
      make_nnet_proto(self.model.get_input_dim(), self.model.get_output_dim(), 
                      nnet_conf, nnet_proto_file)
    elif self.arch == 'lstm':
      make_lstm_proto(self.model.get_input_dim(), self.model.get_output_dim(), 
                      nnet_conf, nnet_proto_file)
    elif self.arch == 'seq2class':
      make_seq2class_proto(self.model.get_input_dim(), self.model.get_output_dim(),
                           nnet_conf, nnet_proto_file)
    else:
      raise RuntimeError("arch type %s not supported", self.arch)


  def __exit__ (self):
    if self.sess is not None:
      self.sess.close()

  
  def read(self, filename):
    filename = filename.strip()

    first_session = True
    if self.sess is not None:
      self.sess.close()
      tf.reset_default_graph()
      first_session = False

    self.graph = tf.Graph()

    with self.graph.as_default():
      self.saver = tf.train.import_meta_graph(filename+'.meta')

    self.model.read_from_file(self.graph, self.use_gpu)

    if first_session:
      self.set_gpu()

    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))

    with self.graph.as_default():
      self.saver.restore(self.sess, filename)


  def write(self, filename):
    with self.graph.as_default():
      save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      saver = tf.train.Saver(save_list, max_to_keep=20)
      saver.save(self.sess, filename)


  def set_gpu(self):
    if self.use_gpu and self.num_gpus != 0:
      p1 = Popen (['pick-gpu', str(self.num_gpus)], stdout=PIPE)
      gpu_ids = str(p1.stdout.read())
      if gpu_ids == "-1":
        if self.wait_gpu:
          logger.info("Waiting for gpus")
          while(gpu_ids == "-1"):
            time.sleep(5)
            p1 = Popen (['pick-gpu', str(self.num_gpus)], stdout=PIPE)
            gpu_ids = str(p1.stdout.read())
        else:
          raise RuntimeError("Picking gpu failed")
      os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ''


  def init_nnet(self, nnet_proto_file, seed = 777):
    self.graph = tf.Graph()

    self.model.init(self.graph, nnet_proto_file, seed)

    self.set_gpu()
    assert self.sess == None
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True))
    self.sess.run(self.model.get_init_all_op())

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()

    return

     
  def init_training(self, optimizer_conf):
    assert self.graph is not None
    self.model.init_training(self.graph, optimizer_conf)
    self.sess.run(self.model.get_init_train_op())

 
  def iter_data(self, logfile, train_gen, learning_rate = None, keep_acc = False, 
                keep_in_prob = 1.0, keep_out_prob = 1.0):
    '''Train/test one iteration; use learning_rate == None to specify test mode'''

    assert self.batch_size*self.num_gpus == train_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    iter_logger.addHandler(fh)

    sum_avg_loss = 0
    sum_accs = 0
    count_steps = 0

    sum_counts = 0        # counts could be frames or utterances
    sum_acc_counts = 0

    start_time = time.time()

    while(True):

      feed_dict, has_data = self.model.prep_feed(train_gen, learning_rate,
                                                 keep_in_prob = keep_in_prob,
                                                 keep_out_prob = keep_out_prob)

      if not has_data:   # no more data for training
        break

      if learning_rate is None:
        loss = self.sess.run(self.model.get_loss(), feed_dict = feed_dict)
      else:
        _, loss = self.sess.run([self.model.get_train_op(), self.model.get_loss()], feed_dict = feed_dict)

      batch_counts = train_gen.get_last_batch_counts()
      sum_avg_loss += loss
      sum_counts += batch_counts
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 10 == 0 or count_steps == 1:
        acc = self.sess.run(self.model.get_eval_acc(), feed_dict = feed_dict)
        sum_accs += 1.0 * acc
        sum_acc_counts += 1.0 * train_gen.get_last_batch_counts()

        # Print status to stdout.
        if count_steps % 10 == 0 or count_steps == 1:
          iter_logger.info("Step %5d: avg loss = %.6f on %d %s (%.2f sec passed, %.2f %s per sec), peek acc: %.2f%%", 
                    count_steps, sum_avg_loss / (count_steps*self.num_gpus), 
                    sum_counts, train_gen.count_units(), duration, sum_counts / duration, 
                    train_gen.count_units(), 100.0*acc/train_gen.get_last_batch_counts())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()

    avg_loss = sum_avg_loss / (count_steps * self.num_gpus)
    if sum_acc_counts == 0:
      avg_acc = None
      avg_acc_str = str(avg_acc)
    else:
      avg_acc = sum_accs/sum_acc_counts
      avg_acc_str = "%.2f%%" % (100.0*avg_acc)

    iter_logger.info("Complete: avg loss = %.6f on %d %s (%.2f sec passed, %.2f %s per sec), peek acc: %s", 
                avg_loss, sum_counts, train_gen.count_units(), duration, 
                sum_counts / duration, train_gen.count_units(), avg_acc_str)

    iter_logger.removeHandler(fh)

    return avg_loss, avg_acc_str


  def patch_to_batches(self, feats):
    ''' patch data so that it matches our batch_size'''
    if len(feats) % self.batch_size == 0:
      return feats
    row2pad = self.batch_size - (len(feats) % self.batch_size)
    feat_dim = len(feats[0])
    feats_padded = np.vstack([feats, np.zeros([row2pad, feat_dim])])
    return feats_padded
 
 
  def pack_utterance(self, feats):
    '''
    args:
      feats: list of array, i.e. matrix of size [num_frames, feat_dim]
    output:
      feat_packs: np 3-d array of size [num_batches, max_length, feat_dim]
      seq_length: np array of size [num_batches]
    '''
    max_length = self.max_length
    jitter_window = self.jitter_window
    start_index = 0
    feats_packed = []
    seq_length = []
    post_pick = []
    pick_start = 0
    pick_end = (max_length + jitter_window) // 2
    while start_index + max_length < len(feats):
      end_index = start_index + max_length
      feats_packed.append(feats[start_index:end_index])
      seq_length.append(max_length)
      post_pick.append([pick_start, pick_end])
      # only the first window starts from 0, all others start from (max_length - jittter_window) / 2
      pick_start = (max_length - jitter_window) // 2      
      start_index += jitter_window

    num_zero = max_length + start_index - len(feats)
    zeros2pad = np.zeros((num_zero, len(feats[0])))
    feats_packed.append(np.concatenate((feats[start_index:], zeros2pad)))
    seq_length.append(len(feats) - start_index)
    # our last window goes till the end of the utterance
    post_pick.append([pick_start, len(feats) - start_index])

    # now we need to pad more zeros to fit the place holder, because each place holder can only host [ batch_size x max_length x feat_dim ] this many data
    baches2pad = self.batch_size - len(feats_packed) % self.batch_size
    if baches2pad != 0:
      zeros2pad = np.zeros((max_length, len(feats[0])))
      for i in range(baches2pad):
        feats_packed.append(zeros2pad)
        seq_length.append(0)
        post_pick.append([0, 0])

    feats_packed = np.array(feats_packed)
    seq_length = np.array(seq_length)

    return feats_packed, seq_length, post_pick


  def predict(self, feats, no_softmax = False):
    if self.arch == 'dnn':
      posts = self.predict_dnn(feats, no_softmax)
    elif self.arch == 'bn':
      posts = self.gen_bn_feats(feats, no_softmax)
    elif self.arch == 'lstm':
      posts = self.predict_lstm(feats, no_softmax)
    else:
      raise RuntimeError("arch type %s not supported", self.arch)
    return posts


  def predict_dnn(self, feats, no_softmax = False):
    '''
    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    posts = []
    num_batches = int(math.ceil(1.0 * len(feats) / self.batch_size))
    for i in range(num_batches):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      # we avoid copying feats, only patch the last batch
      if len(feats) < batch_end:
        feats_padded = self.patch_to_batches(feats[batch_start:,])
      else:
        feats_padded = feats[batch_start:batch_end, :]
      
      feed_dict = self.model.prep_forward_feed(feats_padded)

      if no_softmax:
        batch_posts = self.sess.run(self.model.get_logits(), feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.model.get_outputs(), feed_dict=feed_dict)
      posts.append(batch_posts)

    posts = np.vstack(posts)

    return posts[0:len(feats),:]


  def gen_bn_feats(self):
    '''
    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    bn_outs = []
    num_batches = int(math.ceil(1.0 * len(feats) / self.batch_size))
    for i in range(num_batches):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      # we avoid copying feats, only patch the last batch
      if len(feats) < batch_end:
        feats_padded = self.patch_to_batches(feats[batch_start:,])
      else:
        feats_padded = feats[batch_start:batch_end, :]
      
      feed_dict = self.model.prep_forward_feed(feats_padded)

      batch_bn_outs = self.sess.run(self.model.get_bn(), feed_dict=feed_dict)
      bn_outs.append(batch_bn_outs)

    bn_outs = np.vstack(bn_outs)

    return bn_outs[0:len(feats),:]


  def predict_lstm(self, feats, no_softmax = False):
    '''
    we need a sliding window to generate frame posteriors
    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    # we use a rolling window to process the whole utterance
    feats_packed, seq_length, post_pick = self.pack_utterance(feats)

    posts = []
    assert len(feats_packed) % self.batch_size == 0
    num_batches = int(math.ceil(1.0 * len(feats_packed) / self.batch_size))
    for i in range(num_batches):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      feats_batch = feats_packed[batch_start:batch_end, :]
      seq_length_batch = seq_length[batch_start:batch_end]

      feed_dict = self.model.prep_forward_feed(feats_batch, seq_length_batch, 1.0, 1.0)

      if no_softmax:
        batch_posts = self.sess.run(self.model.get_logits(), feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.model.get_outputs(), feed_dict=feed_dict)
      # batch_posts of size [batch_size, max_len, num_targets]

      for piece in range(self.batch_size):
        if post_pick[piece][0] != post_pick[piece][1]:
          # post_pick specifies the index of posterior to pick out to form decoding sequence
          posts.append(batch_posts[piece, post_pick[piece][0]:post_pick[piece][1]])

    posts = np.concatenate(posts)

    return posts[0:len(feats),:]
