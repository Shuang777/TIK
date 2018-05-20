import tensorflow as tf
import nnet
import make_nnet_proto

class JOINTDNN(object):

  def __init__(self, input_dim, output_dim, batch_size, max_length, 
               num_towers = 1, buckets_tr = None, buckets = None, mode = 'joint'):
    self.type = 'jointdnn'
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.max_length = max_length
    self.num_towers = num_towers
    self.buckets_tr = buckets_tr
    self.buckets_tr = [max_length] if buckets_tr is None else buckets_tr
    self.buckets = [max_length] if buckets is None else buckets
    self.mode = mode   # 'joint', 'asr', 'sid'
    if self.mode in ['joint', 'joint-sid']:
      asr_output_dim, sid_output_dim = output_dim
      self.asr_output_dim = asr_output_dim
      self.sid_output_dim = sid_output_dim

  
  def get_input_dim(self):
    return self.input_dim


  def get_output_dim(self):
    return self.output_dim


  def make_proto(self, nnet_conf, nnet_proto_file):
    if self.mode in ['joint', 'joint-sid']:
      make_nnet_proto.make_nnet_proto(self.input_dim, nnet_conf['hidden_units'],
                                      nnet_conf, nnet_proto_file+'.shared')
      make_nnet_proto.make_asr_proto(nnet_conf['hidden_units'], self.asr_output_dim, 
                                     nnet_conf, nnet_proto_file+'.asr')
      make_nnet_proto.make_sid_proto(nnet_conf['hidden_units'], self.sid_output_dim, 
                                     nnet_conf, nnet_proto_file+'.sid')
    elif self.mode == 'sid':
      make_nnet_proto.sid_append_top(nnet_conf, self.output_dim, nnet_proto_file)


  def init(self, graph, nnet_proto_file, seed = 777):
    if self.num_towers == 1:
      self.init_single(graph, nnet_proto_file, seed)
    else:
      self.init_multi(graph, nnet_proto_file, seed)


  def init_single(self, graph, nnet_proto_file, seed):
    with graph.as_default():
      tf.set_random_seed(seed)
      
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      alpha_holder = tf.placeholder(tf.float32, shape=[], name = 'alpha')
      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('alpha_holder', alpha_holder)
      tf.add_to_collection('beta_holder', beta_holder)

      bucket_tr_feats_holders = []
      bucket_tr_mask_holders = []
      bucket_tr_asr_labels_holders = []
      bucket_tr_sid_labels_holders = []
      bucket_tr_asr_logits = []
      bucket_tr_sid_logits = []
      bucket_tr_asr_outputs = []
      bucket_tr_sid_last_ins = []     # recorded for finetuning phase (we change the last layer)
      for bucket_length in self.buckets_tr:

        reuse = True if bucket_length != self.buckets_tr[0] else False

        feats_holder, mask_holder, \
        asr_labels_holder, sid_labels_holder = nnet.placeholder_jointdnn(self.input_dim, 
                                                             bucket_length,
                                                             self.batch_size,
                                                             name_ext = '_'+str(bucket_length))

        shared_logits = nnet.inference_dnn(feats_holder, nnet_proto_file+'.shared', 
                                           keep_prob_holder, prefix = 'shared_', reuse = reuse)

        asr_logits = nnet.inference_dnn(shared_logits, nnet_proto_file+'.asr', keep_prob_holder, 
                                        prefix = 'asr_', reuse = reuse)

        sid_logits, _, sid_last_layer_in = nnet.inference_seq2class(shared_logits, mask_holder, 
                                                                nnet_proto_file+'.sid', 
                                                                keep_prob_holder, prefix = 'sid_', 
                                                                reuse = reuse)

        asr_outputs = tf.nn.softmax(asr_logits)

        bucket_tr_feats_holders.append(feats_holder)
        bucket_tr_mask_holders.append(mask_holder)
        bucket_tr_asr_labels_holders.append(asr_labels_holder)
        bucket_tr_sid_labels_holders.append(sid_labels_holder)
        bucket_tr_asr_logits.append(asr_logits)
        bucket_tr_sid_logits.append(sid_logits)
        bucket_tr_asr_outputs.append(asr_outputs)
        bucket_tr_sid_last_ins.append(sid_last_layer_in)

        tf.add_to_collection('bucket_tr_feats_holders', feats_holder)
        tf.add_to_collection('bucket_tr_mask_holders', mask_holder)
        tf.add_to_collection('bucket_tr_asr_labels_holders', asr_labels_holder)
        tf.add_to_collection('bucket_tr_sid_labels_holders', sid_labels_holder)
        tf.add_to_collection('bucket_tr_asr_logits', asr_logits)
        tf.add_to_collection('bucket_tr_sid_logits', sid_logits)
        tf.add_to_collection('bucket_tr_asr_outputs', asr_outputs)
        tf.add_to_collection('bucket_tr_sid_last_in', sid_last_layer_in)

      # for decoding buckets
      bucket_feats_holders = []
      bucket_mask_holders = []
      bucket_embeddings = []
      for bucket_length in self.buckets:
        feats_holder, mask_holder, _, _ = nnet.placeholder_jointdnn(self.input_dim, 
                                                                  bucket_length, 1,
                                                                  name_ext = '_'+str(bucket_length))
        shared_logits = nnet.inference_dnn(feats_holder, nnet_proto_file+'.shared', 
                                           keep_prob_holder, prefix = 'shared_', reuse = True)
        _, embeddings, _ = nnet.inference_seq2class(shared_logits, mask_holder, nnet_proto_file+'.sid',
                                           keep_prob_holder, prefix = 'sid_', reuse = True)
        
        bucket_feats_holders.append(feats_holder)
        bucket_mask_holders.append(mask_holder)
        bucket_embeddings.extend(embeddings)    # not a good way to store all the embeddings

        tf.add_to_collection('bucket_feats_holders', feats_holder)
        tf.add_to_collection('bucket_mask_holders', mask_holder)
        for embedding in embeddings:
          tf.add_to_collection('bucket_embeddings', embedding)

      # for training
      self.keep_prob_holder = keep_prob_holder
      self.alpha_holder = alpha_holder
      self.beta_holder = beta_holder
      self.bucket_tr_feats_holders = bucket_tr_feats_holders
      self.bucket_tr_mask_holders = bucket_tr_mask_holders
      self.bucket_tr_asr_labels_holders = bucket_tr_asr_labels_holders
      self.bucket_tr_sid_labels_holders = bucket_tr_sid_labels_holders
      self.bucket_tr_asr_logits = bucket_tr_asr_logits
      self.bucket_tr_sid_logits = bucket_tr_sid_logits
      self.bucket_tr_asr_outputs = bucket_tr_asr_outputs
      self.bucket_tr_sid_last_ins = bucket_tr_sid_last_ins
      # for decoding
      self.bucket_feats_holders = bucket_feats_holders
      self.bucket_mask_holders = bucket_mask_holders
      self.bucket_embeddings = bucket_embeddings
      self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)
      
      self.init_all_op = tf.global_variables_initializer()


  def init_multi(self, graph, nnet_proto_file, seed):
    raise RuntimeError('Not fixed for bucket training yet')
    with graph.as_default(), tf.device('/cpu:0'):
      asr_labels_holder, sid_labels_holder = nnet.placeholder_jointdnn(self.input_dim, 
                                                           self.max_length,
                                                           self.batch_size)
      
      keep_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_prob')
      alpha_holder = tf.placeholder(tf.float32, shape=[], name = 'alpha')
      beta_holder = tf.placeholder(tf.float32, shape=[], name = 'beta')

      shared_logits = nnet.inference_dnn(feats_holder, nnet_proto_file+'.shared', keep_prob_holder,
                                         prefix = 'shared_')

      asr_logits = nnet.inference_dnn(shared_logits, nnet_proto_file+'.asr', keep_prob_holder, 
                                      prefix = 'asr_')
      sid_logits, embeddings = nnet.inference_seq2class(shared_logits, mask_holder, nnet_proto_file+'.sid', 
                                            keep_prob_holder, prefix = 'sid_')

      asr_outputs = tf.nn.softmax(asr_logits)

      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('asr_labels_holder', asr_labels_holder)
      tf.add_to_collection('sid_labels_holder', sid_labels_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_prob_holder', keep_prob_holder)
      tf.add_to_collection('alpha_holder', alpha_holder)
      tf.add_to_collection('beta_holder', beta_holder)
      tf.add_to_collection('asr_logits', asr_logits)
      tf.add_to_collection('sid_logits', sid_logits)
      tf.add_to_collection('asr_outputs', asr_outputs)
      for i in range(len(embeddings)):
        tf.add_to_collection('embeddings', embeddings[i])

      self.feats_holder = feats_holder
      self.asr_labels_holder = asr_labels_holder
      self.sid_labels_holder = sid_labels_holder
      self.mask_holder = mask_holder
      self.keep_prob_holder = keep_prob_holder
      self.alpha_holder = alpha_holder
      self.beta_holder = beta_holder
      self.asr_logits = asr_logits
      self.sid_logits = sid_logits
      self.asr_outputs = asr_outputs
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

            tf.add_to_collection('tower_asr_logits', tower_asr_logits)
            tf.add_to_collection('tower_sid_logits', tower_sid_logits)
            self.tower_asr_logits.append(tower_asr_logits)
            self.tower_sid_logits.append(tower_sid_logits)
            
            tf.add_to_collection('tower_asr_outputs', tower_asr_outputs)
            self.tower_asr_outputs.append(tower_asr_outputs)

      # end towers/gpus
      self.init_all_op = tf.global_variables_initializer()

  
  def finetune_sid(self, graph, nnet_proto_file, seed = 777):
    assert(self.mode == 'sid')
    
    with graph.as_default():
      tf.set_random_seed(seed)

      variables_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

      bucket_tr_sid_logits = tf.get_collection_ref('bucket_tr_sid_logits')
      del bucket_tr_sid_logits[:]

      assert len(self.buckets_tr) == len(self.bucket_tr_sid_last_ins)
      for (bucket_length, layer_in) in zip(self.buckets_tr, self.bucket_tr_sid_last_ins):
        reuse = True if bucket_length != self.buckets_tr[0] else False

        sid_logits = nnet.finetune_sid(layer_in, nnet_proto_file, self.keep_prob_holder, 
                                       prefix = 'sidadd_', reuse = reuse)

        bucket_tr_sid_logits.append(sid_logits)
        # no need to add to collection becasue we use get_collection_ref here

      self.bucket_tr_sid_logits = bucket_tr_sid_logits
 
      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      self.init_additional_op = tf.variables_initializer(new_variables)


  def get_init_all_op(self):
    return self.init_all_op


  def get_init_additional_op(self):
    return self.init_additional_op


  def read_from_file(self, graph, load_towers = False):
    if self.num_towers == 1:
      self.read_single(graph)
    else:
      self.read_multi(graph, load_towers)


  def read_single(self, graph):
    ''' read graph from file '''
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    self.alpha_holder = graph.get_collection('alpha_holder')[0]
    self.beta_holder = graph.get_collection('beta_holder')[0]

    self.bucket_tr_feats_holders = [ x for x in graph.get_collection('bucket_tr_feats_holders') ]
    self.bucket_tr_mask_holders = [ x for x in graph.get_collection('bucket_tr_mask_holders') ]
    self.bucket_tr_asr_labels_holders = [ x for x in graph.get_collection('bucket_tr_asr_labels_holders') ]
    self.bucket_tr_sid_labels_holders = [ x for x in graph.get_collection('bucket_tr_sid_labels_holders') ]
    self.bucket_tr_asr_logits = [ x for x in graph.get_collection('bucket_tr_asr_logits') ]
    self.bucket_tr_sid_logits = [ x for x in graph.get_collection('bucket_tr_sid_logits') ]
    self.bucket_tr_asr_outputs = [ x for x in graph.get_collection('bucket_tr_asr_outputs') ]

    self.bucket_embeddings = [ x for x in graph.get_collection('bucket_embeddings') ]
    self.bucket_feats_holders = [ x for x in graph.get_collection('bucket_feats_holders') ]
    self.bucket_mask_holders = [ x for x in graph.get_collection('bucket_mask_holders') ]
    self.bucket_tr_sid_last_ins = [ x for x in graph.get_collection('bucket_tr_sid_last_in') ]

    self.num_embeddings = len(self.bucket_embeddings) / len(self.bucket_feats_holders)

  
  def read_multi(self, graph, load_towers):
    self.feats_holder = graph.get_collection('feats_holder')[0]
    self.asr_labels_holder = graph.get_collection('asr_labels_holder')[0]
    self.sid_labels_holder = graph.get_collection('sid_labels_holder')[0]
    self.mask_holder = graph.get_collection('mask_holder')[0]
    self.keep_prob_holder = graph.get_collection('keep_prob_holder')[0]
    self.alpha_holder = graph.get_collection('alpha_holder')[0]
    self.beta_holder = graph.get_collection('beta_holder')[0]
    self.asr_logits = graph.get_collection('asr_logits')[0]
    self.asr_outputs = graph.get_collection('asr_outputs')[0]
    self.sid_logits = graph.get_collection('sid_logits')[0]
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
        self.tower_sid_logits.append(tower_sid_logits)


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

      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      if learning_rate is None:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate_holder)
      else:
        opt = nnet.prep_optimizer(optimizer_conf, learning_rate)
    
      self.learning_rate_holder = learning_rate_holder
        
      self.bucket_tr_loss = []
      self.bucket_tr_asr_loss = []
      self.bucket_tr_sid_loss = []
      self.bucket_tr_train_op = []
      self.bucket_tr_asr_train_op = []
      self.bucket_tr_sid_train_op = []
      self.bucket_tr_asr_eval_acc = []
      self.bucket_tr_sid_eval_acc = []

      assert len(self.bucket_tr_asr_logits) == len(self.bucket_tr_sid_logits)
      for (asr_logits, sid_logits, asr_labels_holder, sid_labels_holder, mask_holder) in \
        zip(self.bucket_tr_asr_logits, self.bucket_tr_sid_logits, 
        self.bucket_tr_asr_labels_holders, self.bucket_tr_sid_labels_holders, 
        self.bucket_tr_mask_holders):

        asr_loss = nnet.loss_dnn(asr_logits, asr_labels_holder, mask_holder)
        sid_loss = nnet.loss_dnn(sid_logits, sid_labels_holder)
        loss = self.alpha_holder * asr_loss + self.beta_holder * sid_loss
      
        grads = nnet.get_gradients(opt, loss)
        asr_grads = nnet.get_gradients(opt, asr_loss)
        sid_grads = nnet.get_gradients(opt, sid_loss)
        train_op = nnet.apply_gradients(optimizer_conf, opt, grads)
        asr_train_op = nnet.apply_gradients(optimizer_conf, opt, asr_grads)
        sid_train_op = nnet.apply_gradients(optimizer_conf, opt, sid_grads)

        asr_eval_acc = nnet.evaluation_dnn(asr_logits, asr_labels_holder, mask_holder)
        sid_eval_acc = nnet.evaluation_dnn(sid_logits, sid_labels_holder)
    
        self.bucket_tr_loss.append(loss)
        self.bucket_tr_asr_loss.append(asr_loss)
        self.bucket_tr_sid_loss.append(sid_loss)
        self.bucket_tr_train_op.append(train_op)
        self.bucket_tr_asr_train_op.append(asr_train_op)
        self.bucket_tr_sid_train_op.append(sid_train_op)
        self.bucket_tr_asr_eval_acc.append(asr_eval_acc)
        self.bucket_tr_sid_eval_acc.append(sid_eval_acc)

      variables_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      new_variables = list(set(variables_after) - set(variables_before))
      init_train_op = tf.variables_initializer(new_variables)
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
            loss = self.alpha_holder * asr_loss + self.beta_holder * sid_loss
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

  def set_mode(self, mode):
    assert mode in ['joint', 'asr', 'sid', 'joint-sid']
    self.mode = mode
  
  def get_init_train_op(self):
    return self.init_train_op

  def get_loss(self):
    if self.mode == 'joint':
      return self.bucket_tr_loss[self.last_bucket_id]
    elif self.mode in ['sid', 'joint-sid']:
      return self.bucket_tr_sid_loss[self.last_bucket_id]
    elif self.mode == 'asr':
      return self.bucket_tr_asr_loss[self.last_bucket_id]
    else:
      raise RuntimeError('mode % not supported' % self.mode)

  def get_asr_loss(self):
    return self.bucket_tr_asr_loss[self.last_bucket_id]

  def get_sid_loss(self):
    return self.bucket_tr_sid_loss[self.last_bucket_id]

  def get_eval_acc(self):
    if self.mode == 'asr':
      return self.bucket_tr_asr_eval_acc[self.last_bucket_id]
    elif self.mode == 'sid':
      return self.bucket_tr_sid_eval_acc[self.last_bucket_id]
    else:
      raise RuntimeError('mode % not supported' % self.mode)
  
  def get_asr_eval_acc(self):
    return self.bucket_tr_asr_eval_acc[self.last_bucket_id]

  def get_sid_eval_acc(self):
    return self.bucket_tr_sid_eval_acc[self.last_bucket_id]

  def get_train_op(self):
    if self.mode == 'joint':
      return self.bucket_tr_train_op[self.last_bucket_id]
    elif self.mode == 'asr':
      return self.bucket_tr_asr_train_op[self.last_bucket_id]
    elif self.mode in ['sid', 'joint-sid']:
      return self.bucket_tr_sid_train_op[self.last_bucket_id]

  def get_asr_logits(self):
    return self.bucket_tr_asr_logits[self.last_bucket_id]

  def get_sid_logits(self):
    return self.bucket_tr_sid_logits[self.last_bucket_id]
  
  def get_asr_outputs(self):
    return self.bucket_tr_asr_outputs[self.last_bucket_id]

  def get_embedding(self, index = 0):
    return self.embeddings[index]


  def prep_feed(self, data_gen, train_params):
    if self.mode in ['joint', 'joint-sid']:
      return self.prep_feed_joint(data_gen, train_params)
    elif self.mode == 'sid':
      return self.prep_feed_sid(data_gen, train_params)
    else:
      raise RuntimeError('mode %s not supported yet' % self.mode)


  def prep_feed_joint(self, data_gen, train_params):

    x, y, z, mask, bucket_id = data_gen.get_batch_utterances()

    self.last_bucket_id = bucket_id

    feed_dict = { self.bucket_tr_feats_holders[bucket_id]: x,
                  self.bucket_tr_asr_labels_holders[bucket_id]: y,
                  self.bucket_tr_sid_labels_holders[bucket_id]: z,
                  self.bucket_tr_mask_holders[bucket_id]: mask,
                  self.keep_prob_holder: 1.0,
                  self.alpha_holder: 1.0,
                  self.beta_holder: 0.0} 
    
    if train_params is not None:
      feed_dict.update({
                  self.learning_rate_holder: train_params.get('learning_rate', 0.0),
                  self.keep_prob_holder: train_params.get('keep_prob', 1.0),
                  self.alpha_holder: train_params.get('alpha', 1.0),
                  self.beta_holder: train_params.get('beta', 0.0)})

    return feed_dict, x is not None


  def prep_feed_sid(self, data_gen, params):
    
    x, y, mask, bucket_id = data_gen.get_batch_utterances()

    self.last_bucket_id = bucket_id

    feed_dict = { self.bucket_tr_feats_holders[bucket_id]: x,
                  self.bucket_tr_sid_labels_holders[bucket_id]: y,
                  self.bucket_tr_mask_holders[bucket_id]: mask,
                  self.keep_prob_holder: 1.0,
                  self.alpha_holder: 1.0,
                  self.beta_holder: 0.0} 
    
    if params is not None:
      feed_dict.update({
                  self.learning_rate_holder: params.get('learning_rate', 0.0),
                  self.keep_prob_holder: params.get('keep_prob', 1.0),
                  self.alpha_holder: params.get('alpha', 1.0),
                  self.beta_holder: params.get('beta', 0.0)})

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
