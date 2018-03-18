import tensorflow as tf
import nnet

class BN(object):

  def __init__(self, input_dim, output_dim, batch_size, num_towers = 1):
    self.type = 'bn'
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.num_towers = num_towers


  def get_input_dim(self):
    return self.input_dim


  def get_output_dim(self):
    return self.output_dim

  
  def init(self, graph, nnet_proto_file, seed = 777):

    if self.num_towers == 1:
      self.init_bn_single(graph, nnet_proto_file, seed)
    else:
      self.init_bn_multi(graph, nnet_proto_file, seed)


  def init_bn_single(self, graph, nnet_proto_file, seed = 777):
    ''' initializing nnet from file or config (graph creation) '''

    with graph.as_default():
      tf.set_random_seed(seed)
      feats_holder, labels_holder = nnet.placeholder_dnn(self.input_dim, self.batch_size)
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      
      logits, bn_outputs = nnet.inference_bn(feats_holder, nnet_proto_file, keep_prob_holder)
      outputs = tf.nn.softmax(logits)

      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('bn_outputs', bn_outputs)
    
      self.feats_holder = feats_holder
      self.labels_holder = labels_holder
      self.keep_prob_holder = keep_prob_holder
      self.logits = logits
      self.outputs = outputs
      self.bn_outputs = bn_outputs
      
      self.init_all_op = tf.global_variables_initializer()


  def init_bn_multi(self, graph, nnet_proto_file, seed = 777):
    
    with graph.as_default(), tf.device('/cpu:0'):
      tf.set_random_seed(seed)
      feats_holder, labels_holder = nnet.placeholder_bn(
                                      self.input_dim, 
                                      self.batch_size * self.num_towers)
      
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      
      logits, bn_outputs = nnet.inference_bn(feats_holder, nnet_proto_file, keep_prob_holder)
      outputs = tf.nn.softmax(logits)

      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('bn_outputs', bn_outputs)

      self.feats_holder = feats_holder
      self.labels_holder = labels_holder
      self.keep_prob_holder = keep_prob_holder
      self.logits = logits
      self.outputs = outputs
      self.bn_outputs = bn_outputs

      self.tower_logits = []
      self.tower_outputs = []
      for i in range(self.num_towers):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            
            tower_start_index = i * self.batch_size
            tower_end_index = (i+1) * self.batch_size

            tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:]
            tower_logits, tower_bn = nnet.inference_bn(tower_feats_holder, nnet_proto_file, keep_prob_holder, reuse = True)
            tower_outputs = tf.nn.softmax(tower_logits)

            self.tower_logits.append(tower_logits)
            self.tower_outputs.append(tower_outputs)

            tf.add_to_collection('tower_logits', tower_logits)
            tf.add_to_collection('tower_outputs', tower_outputs)

      # end towers/gpus
      self.init_all_op = tf.global_variables_initializer()
    # end tf graph


  def get_init_all_op(self):
    return self.init_all_op


  def read_from_file(self, graph, load_towers = False):
    if self.num_towers == 1:
      self.read_bn_single(graph)
    else:
      self.read_bn_multi(graph, load_towers)


  def read_bn_single(self, graph):
    ''' read graph from file '''
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.labels_holder = graph.get_collection('labels_holder')[0]
    self.logits = graph.get_collection('logits')[0]
    self.outputs = graph.get_collection('outputs')[0]
    self.bn_outputs = graph.get_collection('bn_outputs')[0]


  def read_bn_multi(self, graph, load_towers):
    ''' read graph from file '''
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.labels_holder = graph.get_collection('labels_holder')[0]
    self.logits = graph.get_collection('logits')[0]
    self.outputs = graph.get_collection('outputs')[0]
    self.bn_outputs = graph.get_collection('bn_outputs')[0]
  
    self.tower_logits = []
    self.tower_outputs = []

    if load_towers:
      for i in range(self.num_towers):
        tower_logits = graph.get_collection('tower_logits')[i]
        tower_outputs = graph.get_collection('tower_outputs')[i]
        self.tower_logits.append(tower_logits)
        self.tower_outputs.append(tower_outputs)


  def init_training(self, graph, optimizer_conf, learning_rate = None):
    if self.num_towers == 1:
      self.init_training_bn_single(graph, optimizer_conf)
    else:
      self.init_training_bn_multi(graph, optimizer_conf)


  def init_training_bn_single(self, graph, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with graph.as_default():

      loss = nnet.loss_dnn(self.logits, self.labels_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation_dnn(self.logits, self.labels_holder)
      
    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc


  def init_training_bn_multi(self, graph, optimizer_conf):
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

            tower_labels_holder = self.labels_holder[tower_start_index:tower_end_index]

            loss = nnet.loss_dnn(self.tower_logits[i], tower_labels_holder)
            tower_losses.append(loss)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
            eval_acc = nnet.evaluation_dnn(self.tower_logits[i], tower_labels_holder)
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


  def prep_feed(self, data_gen, train_params):
    x, y = data_gen.get_batch_frames()

    feed_dict = { self.feats_holder: x,
                  self.labels_holder: y,
                  self.keep_prob_holder: 1.0}

    if train_params is not None:
      feed_dict.update({
                  self.learning_rate_holder: train_params['learning_rate'],
                  self.keep_prob_holder: 1.0})

    return feed_dict, x is not None

  
  def prep_forward_feed(self, x):
    feed_dict = { self.feats_holder: x}
    return feed_dict
  
  
  def get_outputs(self):
    return self.outputs


  def get_logits(self):
    return self.logits


  def get_bn(self):
    return self.bn_outputs
