import shutil
import os
import time
import numpy as np
import tensorflow as tf
from subprocess import Popen,PIPE
import nnet
import math
import logging
from make_nnet_proto import make_nnet_proto, make_lstm_proto

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NNTrainer(object):
  '''
  a class for a neural network that can be used together with Kaldi.
  session is initialized either by read() or by init_nnet().
  '''

  def __init__(self, arch, input_dim, output_dim, batch_size, use_gpu = True, summary_dir = None, max_length = None):
    ''' just some basic config for this trainer '''
    self.arch = arch
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.max_length = max_length    # used for lstm
    self.sess = None
    self.use_gpu = use_gpu
    self.summary_dir = summary_dir


  def make_proto(self, nnet_conf, nnet_proto_file):
    if self.arch == 'dnn':
      make_nnet_proto(self.input_dim, self.output_dim, nnet_conf, nnet_proto_file)
    elif self.arch == 'lstm':
      make_lstm_proto(self.input_dim, self.output_dim, nnet_conf, nnet_proto_file)


  def __exit__ (self):
    if self.sess is not None:
      self.sess.close()

  
  def read(self, filename, num_multi = 0):
    if self.arch == 'dnn':
      self.read_dnn(filename, num_multi)
    elif self.arch == 'lstm':
      self.read_lstm(filename)



  def read_dnn(self, filename, num_multi = 0):
    ''' read graph from file '''
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.saver = tf.train.import_meta_graph(filename+'.meta')
    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
      
    self.logits = self.graph.get_collection('logits')[0]
    self.outputs = self.graph.get_collection('outputs')[0]
    self.num_multi = num_multi
    if num_multi == 0:
      self.feats_holder = self.graph.get_collection('feats_holder')[0]
    elif num_multi > 1:
      self.feats_holder = [self.graph.get_collection('feats_holder'+str(i))[0] for i in range(num_multi)]
      self.switch_holder = [self.graph.get_collection('switch_holder'+str(i))[0] for i in range(num_multi)]

    self.labels_holder = self.graph.get_collection('labels_holder')[0]


  def read_lstm(self, filename):
    ''' read graph from file '''
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.saver = tf.train.import_meta_graph(filename+'.meta')
    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
      
    self.logits = self.graph.get_collection('logits')[0]
    self.outputs = self.graph.get_collection('outputs')[0]
    self.feats_holder = self.graph.get_collection('feats_holder')[0]
    self.labels_holder = self.graph.get_collection('labels_holder')[0]
    self.seq_length_holder = self.graph.get_collection('seq_length_holder')[0]
    self.mask_holder = self.graph.get_collection('mask_holder')[0]


  def write(self, filename):
    with self.graph.as_default():
      save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      saver = tf.train.Saver(save_list, max_to_keep=20)
      saver.save(self.sess, filename)


  def set_gpu(self):
    if self.use_gpu:
      p1 = Popen ('pick-gpu', stdout=PIPE)
      gpu_id = int(p1.stdout.read())
      if gpu_id == -1:
        raise RuntimeError("Unable to pick gpu")
      logging.info("Selecting gpu %d", gpu_id)
      os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ''


  def get_num_subnnets(self):
    return len(self.feats_holder)


  def init_nnet(self, nnet_proto_file, init_file = None):
    if self.arch == 'dnn':
      self.init_dnn(nnet_proto_file, init_file)
    elif self.arch == 'lstm':
      self.init_lstm(nnet_proto_file)


  def init_dnn(self, nnet_proto_file, init_file = None, seed=777):
    ''' initializing nnet from file or config (graph creation) '''
    self.graph = tf.Graph()

    self.num_multi = nnet.scan_subnnet(nnet_proto_file)

    with self.graph.as_default():
      feats_holder, labels_holder = nnet.placeholder_dnn(self.input_dim, self.batch_size, 
                                        multi_subnnet = self.num_multi)
      
      if init_file is not None:
        logits = nnet.inference_from_file(feats_holder, self.input_dim, 
                        self.output_dim, init_file)
      elif self.num_multi == 0:
        logits = nnet.inference_dnn(feats_holder, nnet_proto_file)
      elif self.num_multi > 0:
        self.switch_holder = []
        for i in range(self.num_multi):
          self.switch_holder.append(tf.placeholder(tf.float32, shape=[], name = 'switch'+str(i)))
        logits = nnet.inference_multi(feats_holder, self.switch_holder, nnet_proto_file)
      else:
        raise RuntimeError('')

      outputs = tf.nn.softmax(logits)
      init_all_op = tf.global_variables_initializer()

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      if self.num_multi == 0:
        tf.add_to_collection('feats_holder', feats_holder)
      elif self.num_multi > 0:
        for i in range(self.num_multi):
          tf.add_to_collection('feats_holder'+str(i), feats_holder[i])
          tf.add_to_collection('switch_holder'+str(i), self.switch_holder[i])
      tf.add_to_collection('labels_holder', labels_holder)
    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    tf.set_random_seed(seed)
    self.sess.run(init_all_op)

    self.logits = logits
    self.outputs = outputs
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()


  def init_lstm(self, nnet_proto_file, seed = 777):
    self.graph = tf.Graph()
    with self.graph.as_default():
      feats_holder, seq_length_holder, mask_holder, labels_holder = nnet.placeholder_lstm(self.input_dim, 
                                                                             self.max_length,
                                                                             self.batch_size)

      logits = nnet.inference_lstm(feats_holder, seq_length_holder, nnet_proto_file)

      outputs = tf.nn.softmax(logits)
      init_all_op = tf.global_variables_initializer()

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('seq_length_holder', seq_length_holder)
      tf.add_to_collection('mask_holder', mask_holder)
    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    tf.set_random_seed(seed)
    self.sess.run(init_all_op)

    self.logits = logits
    self.outputs = outputs
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder
    self.seq_length_holder = seq_length_holder
    self.mask_holder = mask_holder

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()


  def init_training(self, optimizer_conf):
    if self.arch == 'dnn':
      self.init_training_dnn(optimizer_conf)
    elif self.arch == 'lstm':
      self.init_training_lstm(optimizer_conf)


  def init_training_dnn(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with self.graph.as_default():

      temp_vars = set(tf.global_variables())
      loss = nnet.loss_dnn(self.logits, self.labels_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      if self.num_multi == 0:
        train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      else:
        train_op = []
        for i in range(self.num_multi):
          train_op.append(nnet.training(optimizer_conf, loss, learning_rate_holder, 
                                        scopes = ['layer0_sub'+ str(i), 'layer_merge', 'layer_shared']))
      eval_acc = nnet.evaluation(self.logits, self.labels_holder)
      if len(set(tf.global_variables()) - temp_vars) != 0:
        # now we need to initialze all variables, even if only newly added one are not initialized
        init_train_op = tf.global_variables_initializer()
        assert self.sess is not None
        self.sess.run(init_train_op)

    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc


  def init_training_lstm(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with self.graph.as_default():

      loss = nnet.loss_lstm(self.logits, self.labels_holder, self.mask_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation_lstm(self.logits, self.labels_holder, self.mask_holder)

    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc

  
  def prepare_feed(self, train_gen, learning_rate):
    if self.arch == 'dnn':
      feed_dict, has_data = self.prepare_feed_dnn(train_gen, learning_rate)
    elif self.arch == 'lstm':
      feed_dict, has_data = self.prepare_feed_lstm(train_gen, learning_rate)

    return feed_dict, has_data


  def prepare_feed_dnn(self, train_gen, learning_rate):

    x, y = train_gen.get_batch_frames()

    feed_dict = { self.feats_holder : x,
                  self.labels_holder : y,
                  self.learning_rate_holder: learning_rate}
    return feed_dict, x is not None


  def prepare_feed_lstm(self, train_gen, learning_rate):
    x, y, seq_length, mask = train_gen.get_batch_utterances()

    feed_dict = { self.feats_holder : x,
                  self.labels_holder : y,
                  self.seq_length_holder: seq_length,
                  self.mask_holder: mask,
                  self.learning_rate_holder: learning_rate}

    return feed_dict, x is not None
    

  def iter_data(self, logfile, train_gen, learning_rate = None, keep_acc = False):
    '''Train/test one iteration; use learning_rate == None to specify test mode'''

    assert self.batch_size == train_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    logger.addHandler(fh)

    sum_avg_loss = 0
    sum_accs = 0
    count_steps = 0

    sum_frames = 0
    sum_acc_frames = 0

    start_time = time.time()

    while(True):

      feed_dict, has_data = self.prepare_feed(train_gen, learning_rate)

      if not has_data:   # no more data for training
        break

      if learning_rate is None:
        loss = self.sess.run(self.loss, feed_dict = feed_dict)
      else:
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)

      batch_frames = train_gen.get_last_batch_frames()
      sum_avg_loss += loss
      sum_frames += batch_frames
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 1000 == 0 or count_steps == 1:
        acc = self.sess.run(self.eval_acc, feed_dict = feed_dict)
        sum_accs += acc
        sum_acc_frames += train_gen.get_last_batch_frames()

        # Print status to stdout.
        if count_steps % 1000 == 0:
          logger.info("Step %5d: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %.2f%%", 
                    count_steps, sum_avg_loss / count_steps, 
                    sum_frames, duration, sum_frames / duration, 
                    100.0*acc/train_gen.get_last_batch_frames())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()

    avg_loss = sum_avg_loss / count_steps
    if sum_acc_frames == 0:
      avg_acc = None
      avg_acc_str = str(avg_acc)
    else:
      avg_acc = sum_accs/sum_acc_frames
      avg_acc_str = "%.2f%%" % (100.0*avg_acc)

    logger.info("Complete: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %s", 
                avg_loss, sum_frames, duration, 
                sum_frames / duration, avg_acc_str)

    logger.removeHandler(fh)

    return avg_loss, avg_acc_str


  def iter_multi_cv(self, logfile, train_gen, learning_rate = None, keep_acc = False):
    '''Train/test one iteration; use learning_rate == None to specify test mode'''

    assert self.batch_size == train_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    logger.addHandler(fh)

    sum_avg_loss = 0
    sum_accs = 0
    count_steps = 0

    sum_frames = 0
    sum_acc_frames = 0

    start_time = time.time()
    while(True):

      x, y = train_gen.get_batch_frames()

      if x is None:   # no more data for training
        break
  
      # fake the second pass
      feed_dict = { self.feats_holder[0]: x,
                    self.feats_holder[1]: x,
                    self.switch_holder[0]: 1,
                    self.switch_holder[1]: 0,
                    self.labels_holder: y,
                    self.learning_rate_holder: learning_rate}

      if learning_rate is None:
        loss = self.sess.run(self.loss, feed_dict = feed_dict)
      else:
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)

      batch_frames = train_gen.get_last_batch_frames()
      sum_avg_loss += loss
      sum_frames += batch_frames
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 1000 == 0 or count_steps == 1:
        acc = self.sess.run(self.eval_acc, feed_dict = feed_dict)
        sum_accs += acc
        sum_acc_frames += batch_frames

        # Print status to stdout.
        logger.info("Step %5d: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %.2f%%", 
                    count_steps, sum_avg_loss / count_steps, 
                    sum_frames, duration, sum_frames / duration, 
                    100.0*acc/train_gen.get_last_batch_frames())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()

    avg_loss = sum_avg_loss / count_steps
    if sum_acc_frames == 0:
      avg_acc = None
      avg_acc_str = str(avg_acc)
    else:
      avg_acc = sum_accs/sum_acc_frames
      avg_acc_str = "%.2f%%" % (100.0*avg_acc)

    logger.info("Complete: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %s", 
                avg_loss, sum_frames, duration, 
                sum_frames / duration, avg_acc_str)

    logger.removeHandler(fh)

    return avg_loss, avg_acc_str


  def iter_multi_data(self, logfile, train_gen, aux_gen, learning_rate, keep_acc = False):

    assert self.batch_size == train_gen.get_batch_size()
    assert self.batch_size == aux_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    logger.addHandler(fh)

    sum_avg_loss = 0
    sum_avg_loss_aux = 0
    sum_accs = 0
    sum_accs_aux = 0
    count_steps = 0

    sum_frames = 0
    sum_acc_frames = 0

    start_time = time.time()
    while(True):

      x, y = train_gen.get_batch_frames()
      x_aux, y_aux = aux_gen.get_batch_frames()

      if x is None:   # no more data for training
        break

      feed_dict = { self.feats_holder[0]: x,
                    self.feats_holder[1]: x_aux,
                    self.switch_holder[0]: 1,
                    self.switch_holder[1]: 0,
                    self.labels_holder: y,
                    self.learning_rate_holder: learning_rate }

      feed_dict_aux = { self.feats_holder[0]: x,
                    self.feats_holder[1]: x_aux,
                    self.switch_holder[0]: 0,
                    self.switch_holder[1]: 1,
                    self.labels_holder: y_aux,
                    self.learning_rate_holder: learning_rate }

      _, loss = self.sess.run([self.train_op[0], self.loss], feed_dict = feed_dict)
      
      _, loss_aux = self.sess.run([self.train_op[1], self.loss], feed_dict = feed_dict_aux)

      batch_frames = train_gen.get_last_batch_frames()
      sum_avg_loss += loss
      sum_avg_loss_aux += loss_aux

      sum_frames += batch_frames
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 1000 == 0 or count_steps == 1:
        acc = self.sess.run(self.eval_acc, feed_dict = feed_dict)
        acc_aux = self.sess.run(self.eval_acc, feed_dict = feed_dict_aux)
        sum_accs += acc
        sum_accs_aux += acc_aux
        sum_acc_frames += train_gen.get_last_batch_frames()

        # Print status to stdout.
        logger.info("Step %5d: avg loss = %.6f (aux loss = %.6f) on %d frames,  (%.2f sec passed, %.2f frames per sec), peek acc: %.2f%% (aux acc: %.2f%%)",
                    count_steps, sum_avg_loss / count_steps, sum_avg_loss_aux / count_steps,
                    sum_frames, duration, sum_frames / duration, 
                    100.0*acc/train_gen.get_last_batch_frames(), 100.0*acc_aux/train_gen.get_batch_size())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()
    aux_gen.reset_batch()

    avg_loss = sum_avg_loss / count_steps
    avg_loss_aux = sum_avg_loss_aux / count_steps
    if sum_acc_frames == 0:
      avg_acc_str = str(None)
      avg_acc_aux_str = str(None)
    else:
      avg_acc = sum_accs/sum_acc_frames
      avg_acc_str = "%.2f%%" % (100.0*avg_acc)
      avg_acc_aux = sum_accs_aux/sum_acc_frames
      avg_acc_aux_str = "%.2f%%" % (100.0*avg_acc_aux)

    logger.info("Complete: avg loss = %.6f (aux loss = %.6f) on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %s (aux acc: %s)", 
                avg_loss, avg_loss_aux, sum_frames, duration, 
                sum_frames / duration, avg_acc_str, avg_acc_aux_str)

    logger.removeHandler(fh)

    return avg_loss, avg_acc_str


  def predict(self, feats, take_log = True):
    if self.arch == 'dnn':
      posts = self.predict_dnn(feats, take_log)
    elif self.arch == 'lstm':
      posts = self.predict_lstm(feats, take_log)

    return posts


  def patch_to_batches(self, feats):
    ''' patch data so that it matches our batch_size'''
    if len(feats) % self.batch_size == 0:
      return feats
    row2pad = self.batch_size - (len(feats) % self.batch_size)
    feat_dim = len(feats[0])
    feats_padded = np.vstack([feats, np.zeros([row2pad, feat_dim])])
    return feats_padded
      

  def predict_dnn(self, feats, take_log = True):
    '''
    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    posts = []
    for i in range(math.ceil(len(feats) / self.batch_size)):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      # we avoid copying feats, only patch the last batch
      if len(feats) < batch_end:
        feats_padded = self.patch_to_batches(feats[batch_start:,])
      else:
        feats_padded = feats[batch_start:batch_end, :]
      
      if self.num_multi == 0:
        feed_dict = {self.feats_holder: feats_padded}
      else:
        feed_dict = {self.feats_holder[0]: feats_padded,
                     self.feats_holder[1]: feats_padded,
                     self.switch_holder[0]: 1,
                     self.switch_holder[1]: 0}

      if take_log:
        batch_posts = self.sess.run(self.outputs, feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.logits, feed_dict=feed_dict)
      posts.append(batch_posts)

    posts = np.vstack(posts)

    return posts[0:len(feats),:]

  
  def pack_utterance(self, feats):
    '''
    args:
      feats: list of array, i.e. matrix of size [num_frames, feat_dim]
    output:
      feat_packs: np 3-d array of size [num_batches, max_length, feat_dim]
      seq_length: np array of size [num_batches]
    '''
    max_length = self.max_length
    start_index = 0
    feats_packed = []
    seq_length = []
    post_pick = []
    pick_start = 0
    while start_index + max_length < len(feats):
      end_index = start_index + max_length
      feats_packed.append(feats[start_index:end_index])
      seq_length.append(max_length)
      post_pick.append([pick_start, (self.max_length + self.out_window) // 2])
      # only the first window starts from 0, all others start from (max_length - out_window) / 2
      pick_start = (self.max_length - self.out_window) // 2      
      start_index += self.out_window

    num_zero = max_length + start_index - len(feats)
    zeros2pad = np.zeros((num_zero, len(feats[0])))
    feats_packed.append(np.concatenate((feats[start_index:], zeros2pad)))
    seq_length.append(len(feats) - start_index)
    # our last window goes till the end of the utterance
    post_pick.append([pick_start, len(feats) - start_index])

    # now we need to pad more zeros to fit the place holder
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


  def predict_lstm(self, feats, take_log = True):
    '''
    we need a sliding window to generate frame posteriors

    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    # for every chunk, we only output these many posteriors, use a rolling window to process the whole utterance
    self.out_window = self.max_length // 2   

    feats_packed, seq_length, post_pick = self.pack_utterance(feats)

    posts = []
    assert len(feats_packed) % self.batch_size == 0
    for i in range(len(feats_packed) // self.batch_size):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      feats_batch = feats_packed[batch_start:batch_end, :]
      seq_length_batch = seq_length[batch_start:batch_end]

      feed_dict = {self.feats_holder: feats_batch,
                   self.seq_length_holder: seq_length_batch}

      if take_log:
        batch_posts = self.sess.run(self.outputs, feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.logits, feed_dict=feed_dict)
      # batch_posts of size [batch_size, max_len, num_targets]

      for piece in range(self.batch_size):
        if post_pick[piece][0] != post_pick[piece][1]:
           posts.append(batch_posts[piece, post_pick[piece][0]:post_pick[piece][1]])

    posts = np.concatenate(posts)

    return posts[0:len(feats),:]
