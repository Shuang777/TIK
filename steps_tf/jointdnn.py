import tensorflow as tf
import nnet
import make_nnet_proto

class JOINTDNN(object):

  def __init__(self, input_dim, output_dim, batch_size, max_length, 
               num_towers = 1, buckets_tr = None, buckets = None):
    self.type = 'jointdnn'
    self.input_dim = input_dim
    self.output_dim = output_dim
    asr_output_dim, sid_output_dim = output_dim
    self.asr_output_dim = asr_output_dim
    self.sid_output_dim = sid_output_dim
    self.batch_size = batch_size
    self.max_length = max_length
    self.num_towers = num_towers
    self.buckets_tr = buckets_tr
    self.buckets = buckets
    if buckets is None:
      self.buckets = [max_length]
  
  def get_input_dim(self):
    return self.input_dim

  def get_output_dim(self):
    return self.output_dim

  def make_proto(self, nnet_conf, nnet_proto_file):
    make_nnet_proto.make_nnet_proto(self.input_dim, nnet_conf['hidden_units'],
                                    nnet_conf, nnet_proto_file+'.shared')
    make_nnet_proto.make_asr_proto(nnet_conf['hidden_units'], self.asr_output_dim, 
                                   nnet_conf, nnet_proto_file+'.asr')
    make_nnet_proto.make_sid_proto(nnet_conf['hidden_units'], self.sid_output_dim, 
                                   nnet_conf, nnet_proto_file+'.sid')

  def init(self, graph, nnet_proto_file, seed = 777):
    if self.num_towers == 1:
      self.init_single(graph, nnet_proto_file, seed)
    else:
      self.init_multi(graph, nnet_proto_file, seed)


  def init_single(self, graph, nnet_proto_file, seed):
    with graph.as_default():
      tf.set_random_seed(seed)
      feats_holder, mask_holder, \
      asr_labels_holder, sid_labels_holder = nnet.placeholder_jointdnn(self.input_dim, 
                                                           self.max_length,
                                                           self.batch_size)
      
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')

      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')

      shared_logits = nnet.inference_dnn(feats_holder, nnet_proto_file+'.shared', keep_prob_holder,
                                         prefix = 'shared_')

      asr_logits = nnet.inference_dnn(shared_logits, nnet_proto_file+'.asr', keep_prob_holder, 
                                      prefix = 'asr_')

      sid_logits, _ = nnet.inference_seq2class(shared_logits, mask_holder, nnet_proto_file+'.sid', 
                                            keep_prob_holder, prefix = 'sid_')

      asr_outputs = tf.nn.softmax(asr_logits)
      sid_outputs = tf.nn.softmax(sid_logits)

      bucket_feats_holders, bucket_mask_holders = nnet.placeholder_uttbuckets(self.input_dim,
                                                                  self.buckets)
      # embeddings stored as [embedding_bucket1_layer1, embedding_bucket1_layer2, 
      #                       embedding_bucket2_layer1, embedding_bucket2_layer2,...]
      bucket_embeddings = []
      for x,y in zip(bucket_feats_holders, bucket_mask_holders):
        shared_logits = nnet.inference_dnn(x, nnet_proto_file+'.shared', keep_prob_holder, 
                                           prefix = 'shared_', reuse = True)
        _, embeddings = nnet.inference_seq2class(shared_logits, y, nnet_proto_file+'.sid',
                                           keep_prob_holder, prefix = 'sid_', reuse = True)
        
        bucket_embeddings.extend(embeddings)    # not a good way to store all the embeddings

      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('asr_labels_holder', asr_labels_holder)
      tf.add_to_collection('sid_labels_holder', sid_labels_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('beta_holder', beta_holder)
      tf.add_to_collection('asr_logits', asr_logits)
      tf.add_to_collection('sid_logits', sid_logits)
      tf.add_to_collection('asr_outputs', asr_outputs)
      tf.add_to_collection('sid_outputs', sid_outputs)
      for x,y in zip(bucket_feats_holders, bucket_mask_holders):
        tf.add_to_collection('bucket_feats_holders', x)
        tf.add_to_collection('bucket_mask_holders', y)
      for embedding in bucket_embeddings:
        tf.add_to_collection('bucket_embeddings', embedding)

      self.feats_holder = feats_holder
      self.asr_labels_holder = asr_labels_holder
      self.sid_labels_holder = sid_labels_holder
      self.mask_holder = mask_holder
      self.keep_prob_holder = keep_prob_holder
      self.beta_holder = beta_holder
      self.asr_logits = asr_logits
      self.sid_logits = sid_logits
      self.asr_outputs = asr_outputs
      self.sid_outputs = sid_outputs
      # bucket related
      self.bucket_feats_holders = bucket_feats_holders
      self.bucket_mask_holders = bucket_mask_holders
      self.bucket_embeddings = bucket_embeddings
      self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)
      
      self.init_all_op = tf.global_variables_initializer()


  def init_multi(self, graph, nnet_proto_file, seed):
    with graph.as_default(), tf.device('/cpu:0'):
      asr_labels_holder, sid_labels_holder = nnet.placeholder_jointdnn(self.input_dim, 
                                                           self.max_length,
                                                           self.batch_size)
      
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')

      shared_logits = nnet.inference_dnn(feats_holder, nnet_proto_file+'.shared', keep_prob_holder,
                                         prefix = 'shared_')

      asr_logits = nnet.inference_dnn(shared_logits, nnet_proto_file+'.asr', keep_prob_holder, 
                                      prefix = 'asr_')
      sid_logits, embeddings = nnet.inference_seq2class(shared_logits, mask_holder, nnet_proto_file+'.sid', 
                                            keep_prob_holder, prefix = 'sid_')

      asr_outputs = tf.nn.softmax(asr_logits)
      sid_outputs = tf.nn.softmax(sid_logits)

      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('asr_labels_holder', asr_labels_holder)
      tf.add_to_collection('sid_labels_holder', sid_labels_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('beta_holder', beta_holder)
      tf.add_to_collection('asr_logits', asr_logits)
      tf.add_to_collection('sid_logits', sid_logits)
      tf.add_to_collection('asr_outputs', asr_outputs)
      tf.add_to_collection('sid_outputs', sid_outputs)
      for i in range(len(embeddings)):
        tf.add_to_collection('embeddings', embeddings[i])

      self.feats_holder = feats_holder
      self.asr_labels_holder = asr_labels_holder
      self.sid_labels_holder = sid_labels_holder
      self.mask_holder = mask_holder
      self.keep_prob_holder = keep_prob_holder
      self.beta_holder = beta_holder
      self.asr_logits = asr_logits
      self.sid_logits = sid_logits
      self.asr_outputs = asr_outputs
      self.sid_outputs = sid_outputs
      self.embeddings = embeddings
      
      self.tower_logits = []
      self.tower_outputs = []
      for i in range(self.num_towers):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            
            tower_start_index = i * self.batch_size
            tower_end_index = (i+1) * self.batch_size

            tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:,:]
            tower_mask_holder = mask_holder[tower_start_index:tower_end_index]

            tower_shared_logits = nnet.inference_dnn(tower_feats_holder, nnet_proto_file+'.shared', 
                                                     keep_prob_holder, prefix = 'shared_', 
                                                     reuse = True)

            tower_asr_logits = nnet.inference_dnn(tower_shared_logits, nnet_proto_file+'.asr', 
                                                  keep_prob_holder, prefix = 'asr_', reuse = True)

            tower_sid_logits, tower_embeddings = nnet.inference_seq2class(tower_shared_logits, 
                                                    tower_mask_holder, nnet_proto_file+'.sid', 
                                                    keep_prob_holder, prefix = 'sid_', reuse = True)

            tower_asr_outputs = tf.nn.softmax(asr_logits)
            tower_sid_outputs = tf.nn.softmax(sid_logits)

            tf.add_to_collection('tower_asr_logits', tower_asr_logits)
            tf.add_to_collection('tower_asr_outputs', tower_asr_outputs)
            self.tower_asr_logits.append(tower_asr_logits)
            self.tower_asr_outputs.append(tower_asr_outputs)
            
            tf.add_to_collection('tower_sid_logits', tower_sid_logits)
            tf.add_to_collection('tower_sid_outputs', tower_sid_outputs)
            self.tower_sid_logits.append(tower_sid_logits)
            self.tower_sid_outputs.append(tower_sid_outputs)

      # end towers/gpus
      self.init_all_op = tf.global_variables_initializer()


  def get_init_all_op(self):
    return self.init_all_op


  def read_from_file(self, graph, load_towers = False):
    if self.num_towers == 1:
      self.read_single(graph)
    else:
      self.read_multi(graph, load_towers)


  def read_single(self, graph):
    ''' read graph from file '''
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.asr_labels_holder = graph.get_collection('asr_labels_holder')[0]
    self.sid_labels_holder = graph.get_collection('sid_labels_holder')[0]
    self.mask_holder = graph.get_collection('mask_holder')[0]
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    if graph.get_collection('beta_holder'):
      self.beta_holder = graph.get_collection('beta_holder')[0]
    else:
      self.beta_holder = None
    self.asr_logits = graph.get_collection('asr_logits')[0]
    self.asr_outputs = graph.get_collection('asr_outputs')[0]
    self.sid_logits = graph.get_collection('sid_logits')[0]
    self.sid_outputs = graph.get_collection('sid_outputs')[0]
    self.bucket_embeddings = []
    self.bucket_feats_holders = []
    self.bucket_mask_holders = []
    for i in range(len(graph.get_collection('bucket_embeddings'))):
      self.bucket_embeddings.append(graph.get_collection('bucket_embeddings')[i])
    for i in range(len(graph.get_collection('bucket_feats_holders'))):
      self.bucket_feats_holders.append(graph.get_collection('bucket_feats_holders')[i])
    for i in range(len(graph.get_collection('bucket_mask_holders'))):
      self.bucket_mask_holders.append(graph.get_collection('bucket_mask_holders')[i])

    self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)

  
  def read_multi(self, graph, load_towers):
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.asr_labels_holder = graph.get_collection('asr_labels_holder')[0]
    self.sid_labels_holder = graph.get_collection('sid_labels_holder')[0]
    self.mask_holder = graph.get_collection('mask_holder')[0]
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    self.beta_holder = graph.get_collection('beta_holder')[0]
    self.asr_logits = graph.get_collection('asr_logits')[0]
    self.asr_outputs = graph.get_collection('asr_outputs')[0]
    self.sid_logits = graph.get_collection('sid_logits')[0]
    self.sid_outputs = graph.get_collection('sid_outputs')[0]
    self.embeddings = []
    for i in range(len(graph.get_collection('embeddings'))):
      self.embeddings.append(graph.get_collection('embeddings')[i])

    self.tower_logits = []
    self.tower_outputs = []

    if load_towers:
      for i in range(self.num_towers):
        tower_asr_logits = graph.get_collection('tower_asr_logits')[i]
        tower_asr_outputs = graph.get_collection('tower_asr_outputs')[i]
        self.tower_asr_logits.append(tower_asr_logits)
        self.tower_asr_outputs.append(tower_asr_outputs)
        
        tower_sid_logits = graph.get_collection('tower_sid_logits')[i]
        tower_sid_outputs = graph.get_collection('tower_sid_outputs')[i]
        self.tower_sid_logits.append(tower_sid_logits)
        self.tower_sid_outputs.append(tower_sid_outputs)


  def init_training(self, graph, optimizer_conf, learning_rate = None):
    if self.num_towers == 1:
      self.init_training_single(graph, optimizer_conf, learning_rate)
    else:
      self.init_training_multi(graph, optimizer_conf, learning_rate)


  def init_training_single(self, graph, optimizer_conf, learning_rate = None):
    ''' initialze training graph; 
    assumes self.asr_logits, self.sid_logits, self.labels_holder in place'''
    with graph.as_default():

      # record variables we have already initialized
      variables_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      asr_loss = nnet.loss_dnn(self.asr_logits, self.asr_labels_holder, self.mask_holder)
      
      sid_loss = nnet.loss_dnn(self.sid_logits, self.sid_labels_holder)

      loss = asr_loss + self.beta_holder * sid_loss

      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      if learning_rate is None:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate_holder)
      else:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate)

      grads = nnet.get_gradients(opt, loss)

      train_op = nnet.apply_gradients(optimizer_conf, opt, grads)

      asr_eval_acc = nnet.evaluation_dnn(self.asr_logits, self.asr_labels_holder, self.mask_holder)
      
      sid_eval_acc = nnet.evaluation_dnn(self.sid_logits, self.sid_labels_holder)
      
      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      init_train_op = tf.variables_initializer(new_variables)

    self.loss = loss
    self.asr_loss = asr_loss
    self.sid_loss = sid_loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.asr_eval_acc = asr_eval_acc
    self.sid_eval_acc = sid_eval_acc

    self.init_train_op = init_train_op

 
  def init_training_multi(self, graph, optimizer_conf):
    tower_losses = []
    tower_grads = []
    tower_asr_accs = []
    tower_sid_accs = []

    with graph.as_default(), tf.device('/cpu:0'):
      
      # record variables we have already initialized
      variables_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      opt = nnet.prep_optimizer(optimizer_conf, learning_rate_holder)

      for i in range(self.num_towers):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % (i)) as scope:
            
            tower_start_index = i*self.batch_size
            tower_end_index = (i+1)*self.batch_size

            tower_asr_labels_holder = self.asr_labels_holder[tower_start_index:tower_end_index]
            tower_sid_labels_holder = self.sid_labels_holder[tower_start_index:tower_end_index]
            tower_mask_holder = self.mask_holder[tower_start_index:tower_end_index]

            asr_loss = nnet.loss_dnn(self.tower_asr_logits[i], tower_asr_labels_holder)
            sid_loss = nnet.loss_dnn(self.tower_sid_logits[i], tower_sid_labels_holder)
            loss = asr_loss + self.beta_holder * sid_loss
            tower_losses.append(loss)

            grads = nnet.get_gradients(opt, loss)
            tower_grads.append(grads)

            asr_eval_acc = nnet.evaluation_dnn(self.tower_asr_logits[i], tower_asr_labels_holder, 
                                               tower_mask_holder)
            sid_eval_acc = nnet.evaluation_dnn(self.tower_sid_logits[i], tower_sid_labels_holder)

            tower_asr_accs.append(asr_eval_acc)
            tower_sid_accs.append(sid_eval_acc)

      grads = nnet.average_gradients(tower_grads)
      train_op = nnet.apply_gradients(optimizer_conf, opt, grads)
      losses = tf.reduce_sum(tower_losses)
      asr_accs = tf.reduce_sum(tower_asr_accs)
      sid_accs = tf.reduce_sum(tower_sid_accs)
      
      # we need to intialize variables that are newly added
      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      init_train_op = tf.variables_initializer(new_variables)

    self.loss = losses
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.asr_eval_acc = asr_accs
    self.sid_eval_acc = sid_accs

    self.init_train_op = init_train_op

  
  def get_init_train_op(self):
    return self.init_train_op

  def get_loss(self):
    return self.loss

  def get_asr_loss(self):
    return self.asr_loss

  def get_sid_loss(self):
    return self.sid_loss

  def get_asr_eval_acc(self):
    return self.asr_eval_acc

  def get_sid_eval_acc(self):
    return self.sid_eval_acc

  def get_train_op(self):
    return self.train_op

  def get_asr_logits(self):
    return self.asr_logits

  def get_sid_logits(self):
    return self.sid_logits
  
  def get_asr_outputs(self):
    return self.asr_outputs

  def get_sid_outputs(self):
    return self.sid_outputs

  def get_embedding(self, index = 0):
    return self.embeddings[index]


  def prep_feed(self, data_gen, train_params):

    x, y, z, mask, bucket_id = data_gen.get_batch_utterances()

    feed_dict = { self.feats_holder: x,
                  self.asr_labels_holder: y,
                  self.sid_labels_holder: z,
                  self.mask_holder: mask,
                  self.keep_prob_holder: 1.0,
                  self.beta_holder: 0.0} 
    
    if train_params is not None:
      feed_dict.update({
                  self.learning_rate_holder: train_params.get('learning_rate', 0.0),
                  self.keep_prob_holder: train_params.get('keep_prob', 1.0),
                  self.beta_holder: train_params.get('beta', 0.0)})

    return feed_dict, x is not None

  def prep_forward_feed(self, x):

    feed_dict = { self.feats_holder: x}

    return feed_dict
    
  def prep_forward_sid(self, x, mask, bucket_id, embedding_index):
    bucket_feats_holder = self.bucket_feats_holders[bucket_id]
    bucket_mask_holder = self.bucket_mask_holders[bucket_id]
    embedding = self.bucket_embeddings[self.num_embeddings * bucket_id + embedding_index]

    feed_dict = { bucket_feats_holder: x,
                  bucket_mask_holder: mask}

    return feed_dict, embedding
