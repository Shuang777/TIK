import tensorflow as tf
import nnet
import make_nnet_proto

class SEQ2CLASS(object):

  def __init__(self, input_dim, output_dim, batch_size, max_length, 
               num_towers = 1, buckets_tr = None, buckets = None):
    self.type = 'lstm'
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.max_length = max_length
    self.num_towers = num_towers
    self.buckets_tr = [max_length] if buckets_tr is None else buckets_tr
    self.buckets = [max_length] if buckets is None else buckets

  
  def get_input_dim(self):
    return self.input_dim


  def get_output_dim(self):
    return self.output_dim
  
  
  def make_proto(self, nnet_conf, nnet_proto_file):
    make_nnet_proto.make_seq2class_proto(self.input_dim, self.output_dim, 
                                         nnet_conf, nnet_proto_file)


  def init(self, graph, nnet_proto_file, seed = 777):
    if self.num_towers == 1:
      self.init_single(graph, nnet_proto_file, seed)
    else:
      self.init_multi(graph, nnet_proto_file, seed)


  def init_single(self, graph, nnet_proto_file, seed):
    with graph.as_default():
      tf.set_random_seed(seed)

      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')
      
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('beta_holder', beta_holder)

      bucket_tr_feats_holders = []
      bucket_tr_mask_holders = []
      bucket_tr_labels_holders = []
      bucket_tr_logits = []
      bucket_tr_outputs = []
      # for training buckets
      for bucket_length in self.buckets_tr:
        
        reuse = True if bucket_length != self.buckets_tr[0] else False

        feats_holder, mask_holder, labels_holder = nnet.placeholder_seq2class(self.input_dim, 
                                                           bucket_length,
                                                           self.batch_size,
                                                           name_ext = '_tr'+str(bucket_length))

        logits, _, _ = nnet.inference_seq2class(feats_holder, mask_holder, nnet_proto_file, 
                                             keep_prob_holder, reuse = reuse)

        outputs = tf.nn.softmax(logits)

        bucket_tr_feats_holders.append(feats_holder)
        bucket_tr_mask_holders.append(mask_holder)
        bucket_tr_labels_holders.append(labels_holder)
        bucket_tr_logits.append(logits)
        bucket_tr_outputs.append(outputs)

        tf.add_to_collection('bucket_tr_feats_holders', feats_holder)
        tf.add_to_collection('bucket_tr_labels_holders', labels_holder)
        tf.add_to_collection('bucket_tr_mask_holders', mask_holder)
        tf.add_to_collection('bucket_tr_logits', logits)
        tf.add_to_collection('bucket_tr_outputs', outputs)


      # for decoding buckets
      bucket_feats_holders = []
      bucket_mask_holders = []
      bucket_label_holders = []
      bucket_embeddings = []
      for bucket_length in self.buckets:
        feats_holder, mask_holder, _ = nnet.placeholder_seq2class(self.input_dim,
                                                                  bucket_length, 1,
                                                                  name_ext = '_'+str(bucket_length))
        _, embeddings, _ = nnet.inference_seq2class(feats_holder, mask_holder, 
                                                 nnet_proto_file, keep_prob_holder, reuse = True)

        bucket_feats_holders.append(feats_holder)
        bucket_mask_holders.append(mask_holder)
        # note that each bucket_length may have multiple embeddings
        bucket_embeddings.extend(embeddings)

        tf.add_to_collection('bucket_feats_holders', feats_holder)
        tf.add_to_collection('bucket_mask_holders', mask_holder)
        for i in range(len(embeddings)):
          tf.add_to_collection('bucket_embeddings', embeddings[i])

      self.keep_prob_holder = keep_prob_holder
      self.beta_holder = beta_holder

      self.bucket_tr_feats_holders = bucket_tr_feats_holders
      self.bucket_tr_labels_holders = bucket_tr_labels_holders
      self.bucket_tr_mask_holders = bucket_tr_mask_holders
      self.bucket_tr_logits = bucket_tr_logits
      self.bucket_tr_outputs = bucket_tr_outputs
  
      self.bucket_feats_holders = bucket_feats_holders
      self.bucket_mask_holders = bucket_mask_holders
      self.bucket_embeddings = bucket_embeddings
    
      self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)
      
      self.init_all_op = tf.global_variables_initializer()


  def init_multi(self, graph, nnet_proto_file, seed):
    with graph.as_default(), tf.device('/cpu:0'):
      tf.set_random_seed(seed)

      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')
      
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('beta_holder', beta_holder)

      # for training buckets 
      bucket_tr_feats_holders = []
      bucket_tr_mask_holders = []
      bucket_tr_labels_holders = []
      bucket_tr_tower_logits = []
      bucket_tr_tower_outputs = []
      for bucket_length in self.buckets_tr:

        feats_holder, mask_holder, labels_holder = nnet.placeholder_seq2class(self.input_dim, 
                                                           bucket_length,
                                                           self.batch_size*self.num_towers,
                                                           name_ext = '_tr'+str(bucket_length))
      
        bucket_tr_feats_holders.append(feats_holder)
        bucket_tr_mask_holders.append(mask_holder)
        bucket_tr_labels_holders.append(labels_holder)

        tf.add_to_collection('bucket_tr_feats_holders', feats_holder)
        tf.add_to_collection('bucket_tr_mask_holders', mask_holder)
        tf.add_to_collection('bucket_tr_labels_holders', labels_holder)

        tower_logits = []
        tower_outputs = []
        for i in range(self.num_towers):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % i) as scope:

              reuse = False if bucket_length == self.buckets_tr[0] and i == 0 else True
              
              tower_start_index = i * self.batch_size
              tower_end_index = (i+1) * self.batch_size

              tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:,:]
              tower_mask_holder = mask_holder[tower_start_index:tower_end_index]

              tower_logit, _, _ = nnet.inference_seq2class(tower_feats_holder,
                                                tower_mask_holder, nnet_proto_file, 
                                                keep_prob_holder, reuse = reuse)

              tower_output = tf.nn.softmax(tower_logit)

              tf.add_to_collection('bucket_tr%d_tower_logits' % bucket_length, tower_logit)
              tf.add_to_collection('bucket_tr%d_tower_outputs' % bucket_length, tower_output)

              tower_logits.append(tower_logit)
              tower_outputs.append(tower_output)

        bucket_tr_tower_logits.append(tower_logits)
        bucket_tr_tower_outputs.append(tower_outputs)

      # for decoding buckets
      bucket_feats_holders = []
      bucket_mask_holders = []
      bucket_label_holders = []
      bucket_embeddings = []
      for bucket_length in self.buckets:

        feats_holder, mask_holder, _ = nnet.placeholder_seq2class(self.input_dim,
                                                                  bucket_length, 1,
                                                                  name_ext = '_'+str(bucket_length))
        _, embeddings, _ = nnet.inference_seq2class(feats_holder, mask_holder, 
                                                 nnet_proto_file, keep_prob_holder, reuse = True)

        bucket_feats_holders.append(feats_holder)
        bucket_mask_holders.append(mask_holder)
        # note that each bucket_length may have multiple embeddings
        bucket_embeddings.extend(embeddings)

        tf.add_to_collection('bucket_feats_holders', feats_holder)
        tf.add_to_collection('bucket_mask_holders', mask_holder)
        for i in range(len(embeddings)):
          tf.add_to_collection('bucket_embeddings', embeddings[i])

      self.keep_prob_holder = keep_prob_holder
      self.beta_holder = beta_holder

      self.bucket_tr_feats_holders = bucket_tr_feats_holders
      self.bucket_tr_mask_holders = bucket_tr_mask_holders
      self.bucket_tr_labels_holders = bucket_tr_labels_holders
      self.bucket_tr_tower_logits = bucket_tr_tower_logits
      self.bucket_tr_tower_outputs = bucket_tr_tower_outputs
      
      self.bucket_feats_holders = bucket_feats_holders
      self.bucket_mask_holders = bucket_mask_holders
      self.bucket_embeddings = bucket_embeddings

      self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)

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
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    self.beta_holder = graph.get_collection('beta_holder')[0] if \
                            graph.get_collection('beta_holder') else None
    
    self.bucket_tr_feats_holders = [ x for x in graph.get_collection('bucket_tr_feats_holders') ]
    self.bucket_tr_mask_holders = [ x for x in graph.get_collection('bucket_tr_mask_holders') ]
    self.bucket_tr_labels_holders = [ x for x in graph.get_collection('bucket_tr_labels_holders') ]
    self.bucket_tr_logits = [ x for x in graph.get_collection('bucket_tr_logits') ]
    self.bucket_tr_outputs = [ x for x in graph.get_collection('bucket_tr_outputs') ]

    self.bucket_feats_holders = [ x for x in graph.get_collection('bucket_feats_holders') ]
    self.bucket_mask_holders = [ x for x in graph.get_collection('bucket_mask_holders') ]
    self.bucket_embeddings = [ x for x in graph.get_collection('bucket_embeddings') ]
    if len(self.bucket_embeddings) == 0:
      # to keep compatible with previous format
      self.bucket_embeddings = [ x for x in graph.get_collection('embeddings') ]

    self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)

  
  def read_multi(self, graph, load_towers):
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    self.beta_holder = graph.get_collection('beta_holder')[0] if \
                            graph.get_collection('beta_holder') else None

    self.bucket_tr_feats_holders = [ x for x in graph.get_collection('bucket_tr_feats_holders') ]
    self.bucket_tr_mask_holders = [ x for x in graph.get_collection('bucket_tr_mask_holders') ]
    self.bucket_tr_labels_holders = [ x for x in graph.get_collection('bucket_tr_labels_holders') ]
    self.bucket_tr_tower_logits = []
    self.bucket_tr_tower_outputs = []
    if load_towers:
      for bucket_length in self.buckets_tr:
        tower_logits = [x for x in graph.get_collection('bucket_tr%d_tower_logits'%bucket_length)]
        tower_outputs = [x for x in graph.get_collection('bucket_tr%d_tower_outputs'%bucket_length)]
        assert len(tower_logits) == self.num_towers
        assert len(tower_outputs) == self.num_towers
        self.bucket_tr_tower_logits.append(tower_logits)
        self.bucket_tr_tower_outputs.append(tower_outputs)

    self.bucket_feats_holders = [ x for x in graph.get_collection('bucket_feats_holders') ]
    self.bucket_mask_holders = [ x for x in graph.get_collection('bucket_mask_holders') ]
    self.bucket_embeddings = [ x for x in graph.get_collection('bucket_embeddings') ]
    if len(self.bucket_embeddings) == 0:
      # to keep compatible with previous format
      self.bucket_embeddings = [ x for x in graph.get_collection('embeddings') ]
    
    self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)


  def init_training(self, graph, optimizer_conf, learning_rate = None):
    if self.num_towers == 1:
      self.init_training_single(graph, optimizer_conf, learning_rate)
    else:
      self.init_training_multi(graph, optimizer_conf, learning_rate)


  def init_training_single(self, graph, optimizer_conf, learning_rate = None):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with graph.as_default():

      # record variables we have already initialized
      variables_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      if learning_rate is None:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate_holder)
      else:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate)

      self.learning_rate_holder = learning_rate_holder

      #regularizer = tf.contrib.layers.l2_regularizer(scale = self.beta_holder)
      #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      #reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

      self.bucket_tr_loss = []
      self.bucket_tr_train_op = []
      self.bucket_tr_eval_acc = []
      for (logits, labels_holder) in zip(self.bucket_tr_logits, self.bucket_tr_labels_holders):
        loss = nnet.loss_dnn(logits, labels_holder)
        # add reguarlization
        #loss += reg_term

        grads = nnet.get_gradients(opt, loss)
        train_op = nnet.apply_gradients(optimizer_conf, opt, grads)
        eval_acc = nnet.evaluation_dnn(logits, labels_holder)

        self.bucket_tr_loss.append(loss)
        self.bucket_tr_train_op.append(train_op)
        self.bucket_tr_eval_acc.append(eval_acc)
      
      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      init_train_op = tf.variables_initializer(new_variables)
      self.init_train_op = init_train_op

 
  def init_training_multi(self, graph, optimizer_conf, learning_rate = None):

    self.bucket_tr_loss = []
    self.bucket_tr_train_op = []
    self.bucket_tr_eval_acc = []

    with graph.as_default(), tf.device('/cpu:0'):

      # record variables we have already initialized
      variables_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      if learning_rate is None:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate_holder)
      else:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate)

      self.learning_rate_holder = learning_rate_holder

      for (tower_logits, tower_outputs, labels_holder) in zip(self.bucket_tr_tower_logits,
                                                              self.bucket_tr_tower_outputs,
                                                              self.bucket_tr_labels_holders):
        tower_losses = []
        tower_grads = []
        tower_accs = []
        for i in range(self.num_towers):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:
              
              tower_start_index = i*self.batch_size
              tower_end_index = (i+1)*self.batch_size

              tower_labels_holder = labels_holder[tower_start_index:tower_end_index]

              loss = nnet.loss_dnn(tower_logits[i], tower_labels_holder)
              tower_losses.append(loss)
              grads = nnet.get_gradients(opt, loss)
              tower_grads.append(grads)
              eval_acc = nnet.evaluation_dnn(tower_logits[i], tower_labels_holder)
              tower_accs.append(eval_acc)

        bucket_tr_grads = nnet.average_gradients(tower_grads)
        bucket_tr_train_op = nnet.apply_gradients(optimizer_conf, opt, bucket_tr_grads)
        bucket_tr_loss = tf.reduce_sum(tower_losses)
        bucket_tr_eval_acc = tf.reduce_sum(tower_accs)

        self.bucket_tr_loss.append(bucket_tr_loss)
        self.bucket_tr_train_op.append(bucket_tr_train_op)
        self.bucket_tr_eval_acc.append(bucket_tr_eval_acc)
        
      # we need to intialize variables that are newly added
      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      init_train_op = tf.variables_initializer(new_variables)
      self.init_train_op = init_train_op

  
  def get_init_train_op(self):
    return self.init_train_op


  def get_loss(self):
    return self.bucket_tr_loss[self.last_bucket_id]

  
  def get_eval_acc(self):
    return self.bucket_tr_eval_acc[self.last_bucket_id]


  def get_train_op(self):
    return self.bucket_tr_train_op[self.last_bucket_id]


  def get_logits(self):
    return self.bucket_tr_logits[self.last_bucket_id]

  
  def get_outputs(self):
    return self.bucket_tr_outputs[self.last_bucket_id]


  def get_embedding(self, index = 0):
    return self.embeddings[index]


  def prep_feed(self, data_gen, params):

    x, y, mask, bucket_id = data_gen.get_batch_utterances()

    self.last_bucket_id = bucket_id

    feed_dict = { self.bucket_tr_feats_holders[bucket_id]: x,
                  self.bucket_tr_labels_holders[bucket_id]: y,
                  self.bucket_tr_mask_holders[bucket_id]: mask,
                  self.keep_prob_holder: 1.0,
                  self.beta_holder: 0.0} 
    
    if params is not None:
      feed_dict.update({
                  self.learning_rate_holder: params.get('learning_rate', 0.0),
                  self.keep_prob_holder: params.get('keep_prob', 1.0),
                  self.beta_holder: params.get('beta', 0.0)})

    return feed_dict, x is not None


  def prep_forward_feed(self, x, mask, bucket_id, embedding_index):
    bucket_feats_holder = self.bucket_feats_holders[bucket_id]
    bucket_mask_holder = self.bucket_mask_holders[bucket_id]
    embedding = self.bucket_embeddings[self.num_embeddings * bucket_id + embedding_index]

    feed_dict = { bucket_feats_holder: x,
                  bucket_mask_holder: mask,
                  self.keep_prob_holder: 1.0,
                  self.beta_holder: 0.0}

    return feed_dict, embedding
