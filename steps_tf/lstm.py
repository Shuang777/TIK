import tensorflow as tf
import nnet

class LSTM(object):

  def __init__(self, input_dim, output_dim, batch_size, max_length, num_towers = 1):
    self.type = 'lstm'
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.max_length = max_length
    self.num_towers = num_towers

  
  def get_input_dim(self):
    return self.input_dim


  def get_output_dim(self):
    return self.output_dim


  def init(self, graph, nnet_proto_file, seed = 777):
    if self.num_towers == 1:
      self.init_lstm_single(graph, nnet_proto_file, seed)
    else:
      self.init_lstm_multi(graph, nnet_proto_file, seed)


  def init_lstm_single(self, graph, nnet_proto_file, seed):
    with graph.as_default():
      feats_holder, seq_length_holder, \
        mask_holder, labels_holder = nnet.placeholder_lstm(self.input_dim, 
                                                           self.max_length,
                                                           self.batch_size)
      
      keep_in_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_in_prob')
      keep_out_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_out_prob')

      logits = nnet.inference_lstm(feats_holder, seq_length_holder, nnet_proto_file, 
                                   keep_in_prob_holder, keep_out_prob_holder)

      outputs = tf.nn.softmax(logits)

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('seq_length_holder', seq_length_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_in_prob_holder', keep_in_prob_holder)
      tf.add_to_collection('keep_out_prob_holder', keep_out_prob_holder)

      self.feats_holder = feats_holder
      self.labels_holder = labels_holder
      self.seq_length_holder = seq_length_holder
      self.mask_holder = mask_holder
      self.keep_in_prob_holder = keep_in_prob_holder
      self.keep_out_prob_holder = keep_out_prob_holder
      self.logits = logits
      self.outputs = outputs
      
      self.init_all_op = tf.global_variables_initializer()


  def init_lstm_multi(self, graph, nnet_proto_file, seed):
    with graph.as_default(), tf.device('/cpu:0'):

      feats_holder, seq_length_holder, \
        mask_holder, labels_holder = nnet.placeholder_lstm(
                                         self.input_dim,
                                         self.max_length,
                                         self.batch_size*self.num_towers)
      
      keep_in_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_in_prob')
      keep_out_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_out_prob')

      logits = nnet.inference_lstm(feats_holder, seq_length_holder, nnet_proto_file,
                                   keep_in_prob_holder, keep_out_prob_holder)

      outputs = tf.nn.softmax(logits)

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('seq_length_holder', seq_length_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_in_prob_holder', keep_in_prob_holder)
      tf.add_to_collection('keep_out_prob_holder', keep_out_prob_holder)

      self.feats_holder = feats_holder
      self.labels_holder = labels_holder
      self.seq_length_holder = seq_length_holder
      self.mask_holder = mask_holder
      self.keep_in_prob_holder = keep_in_prob_holder
      self.keep_out_prob_holder = keep_out_prob_holder
      self.logits = logits
      self.outputs = outputs

      self.tower_logits = []
      self.tower_outputs = []
      for i in range(self.num_towers):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            
            tower_start_index = i * self.batch_size
            tower_end_index = (i+1) * self.batch_size

            tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:,:]
            tower_seq_length_holder = seq_length_holder[tower_start_index:tower_end_index]

            tower_logits = nnet.inference_lstm(tower_feats_holder, tower_seq_length_holder, 
                                               nnet_proto_file, keep_in_prob_holder, 
                                               keep_out_prob_holder, reuse = True)

            tower_outputs = tf.nn.softmax(tower_logits)

            tf.add_to_collection('tower_logits', tower_logits)
            tf.add_to_collection('tower_outputs', tower_outputs)
            self.tower_logits.append(tower_logits)
            self.tower_outputs.append(tower_outputs)

      # end towers/gpus
      self.init_all_op = tf.global_variables_initializer()


  def get_init_all_op(self):
    return self.init_all_op


  def read_from_file(self, graph, load_towers = False):
    if self.num_towers == 1:
      self.read_lstm_single(graph)
    else:
      self.read_lstm_multi(graph, load_towers)


  def read_lstm_single(self, graph):
    ''' read graph from file '''
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.labels_holder = graph.get_collection('labels_holder')[0]
    self.seq_length_holder = graph.get_collection('seq_length_holder')[0]
    self.mask_holder = graph.get_collection('mask_holder')[0]
    self.keep_in_prob_holder = graph.get_collection('keep_in_prob_holder')[0]
    self.keep_out_prob_holder = graph.get_collection('keep_out_prob_holder')[0]
    self.logits = graph.get_collection('logits')[0]
    self.outputs = graph.get_collection('outputs')[0]

  
  def read_lstm_multi(self, graph, load_towers):
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.labels_holder = graph.get_collection('labels_holder')[0]
    self.seq_length_holder = graph.get_collection('seq_length_holder')[0]
    self.mask_holder = graph.get_collection('mask_holder')[0]
    self.keep_in_prob_holder = graph.get_collection('keep_in_prob_holder')[0]
    self.keep_out_prob_holder = graph.get_collection('keep_out_prob_holder')[0]
    self.logits = graph.get_collection('logits')[0]
    self.outputs = graph.get_collection('outputs')[0]

    self.tower_logits = []
    self.tower_outputs = []

    if load_towers:
      for i in range(self.num_towers):
        tower_logits = graph.get_collection('tower_logits')[i]
        tower_outputs = graph.get_collection('tower_outputs')[i]
        self.tower_logits.append(tower_logits)
        self.tower_outputs.append(tower_outputs)


  def init_training(self, graph, optimizer_conf):
    if self.num_towers == 1:
      self.init_training_lstm_single(graph, optimizer_conf)
    else:
      self.init_training_lstm_multi(graph, optimizer_conf)


  def init_training_lstm_single(self, graph, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with graph.as_default():

      loss = nnet.loss_lstm(self.logits, self.labels_holder, self.mask_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation_lstm(self.logits, self.labels_holder, self.mask_holder)

    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc

 
  def init_training_lstm_multi(self, graph, optimizer_conf):
    tower_losses = []
    tower_grads = []
    tower_accs = []

    with graph.as_default(), tf.device('/cpu:0'):
      
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      assert optimizer_conf['op_type'].lower() == 'sgd'
      opt = tf.train.GradientDescentOptimizer(learning_rate_holder)

      for i in range(self.num_towers):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % (i)) as scope:
            
            tower_start_index = i*self.batch_size
            tower_end_index = (i+1)*self.batch_size

            tower_mask_holder = self.mask_holder[tower_start_index:tower_end_index,:]
            tower_labels_holder = self.labels_holder[tower_start_index:tower_end_index,:]

            loss = nnet.loss_lstm(self.tower_logits[i], tower_labels_holder, tower_mask_holder)
            tower_losses.append(loss)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
            eval_acc = nnet.evaluation_lstm(self.tower_logits[i], tower_labels_holder, tower_mask_holder)
            tower_accs.append(eval_acc)

      grads = nnet.average_gradients(tower_grads)
      train_op = opt.apply_gradients(grads)
      losses = tf.reduce_sum(tower_losses)
      accs = tf.reduce_sum(tower_accs)

    self.loss = losses
    self.eval_acc = accs
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op


  def get_loss(self):
    return self.loss


  def get_eval_acc(self):
    return self.eval_acc


  def get_train_op(self):
    return self.train_op


  def prep_feed(self, x, y, seq_length, mask, 
                learning_rate, keep_in_prob, keep_out_prob):

    feed_dict = { self.feats_holder: x,
                  self.labels_holder: y,
                  self.seq_length_holder: seq_length,
                  self.mask_holder: mask,
                  self.learning_rate_holder: learning_rate,
                  self.keep_in_prob_holder: keep_in_prob,
                  self.keep_out_prob_holder: keep_out_prob}

    return feed_dict


  def prep_forward_feed(self, x, seq_length, keep_in_prob, keep_out_prob):

    feed_dict = { self.feats_holder: x,
                  self.seq_length_holder: seq_length,
                  self.keep_in_prob_holder: keep_in_prob,
                  self.keep_out_prob_holder: keep_out_prob }

    return feed_dict
