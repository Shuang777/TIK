import tensorflow as tf
import math
import layer


def placeholder_dnn(input_dim, batch_size, multi_subnnet = 0):
  if multi_subnnet == 0:
    feats_holder = tf.placeholder(tf.float32, shape=(batch_size, input_dim), name='feature')
  else:
    feats_holder = []
    for i in range(multi_subnnet):
      feats_holder.append(tf.placeholder(tf.float32, shape=(batch_size, input_dim), name='feature'+str(i)))

  labels_holder = tf.placeholder(tf.int32, shape=(batch_size), name='target')
  return feats_holder, labels_holder


def placeholder_lstm(input_dim, max_length, batch_size):
  '''
  outputs:
    feats_holder, labels_holder, seq_length_holder, mask_holder
  '''
  feats_holder = tf.placeholder(tf.float32, shape=(batch_size, max_length, input_dim), name='feature')

  labels_holder = tf.placeholder(tf.int32, shape=(batch_size, max_length), name='target')

  seq_length_holder = tf.placeholder(tf.int32, shape=(batch_size), name='seq_length')

  mask_holder = tf.placeholder(tf.float32, shape=(batch_size, max_length), name='mask')

  return feats_holder, seq_length_holder, mask_holder, labels_holder


def inference_dnn(feats_holder, nnet_proto_file, reuse = False):
  '''
  args:
    feats_holder: np 2-d array of size [num_batch, feat_dim]
  outputs: 
    logits: np 2-d array of size [num_batch, num_targets]
  '''
  
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  layer_in = feats_holder
  count_layer = 1
  for line in nnet_proto:
    line = line.strip()
    if line == '</NnetProto>':
      break
    with tf.variable_scope('layer'+str(count_layer), reuse = reuse):
      layer_out = build_layer(line, layer_in, reuse = reuse)
      layer_in = layer_out
      count_layer += 1
  logits = layer_out
  return logits


def inference_bn(feats_holder, nnet_proto_file, reuse = False):
  '''
  args:
    feats_holder: np 2-d array of size [num_batch, feat_dim]
  outputs: 
    logits: np 2-d array of size [num_batch, num_targets]
  '''
  
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  layer_in = feats_holder
  count_layer = 1
  is_bn_layer = False
  for line in nnet_proto:
    line = line.strip()
    if line == '</NnetProto>':
      break
    elif line == '<BottleNeck>':
      is_bn_layer = True
      continue
    elif line == '</BottleNeck>':
      is_bn_layer = False
      continue
    with tf.variable_scope('layer'+str(count_layer), reuse = reuse):
      layer_out = build_layer(line, layer_in, reuse = reuse)
      layer_in = layer_out
      count_layer += 1
      if is_bn_layer:
        bn_out = layer_out
  logits = layer_out
  return logits, bn_out


def scan_subnnet(nnet_proto_file):
  nnet_proto = open(nnet_proto_file, 'r')
  count_subnnet = 0
  for line in nnet_proto:
    line = line.strip()
    if line == '<SubNnet>':
      count_subnnet += 1
  return count_subnnet


def inference_lstm(feats_holder, seq_length_holder, nnet_proto_file, 
                   keep_in_prob_holder, keep_out_prob_holder, reuse = False):
  '''
  args:
    feats_holder: np 3-d array of size [num_batch, max_length, feat_dim]
    seq_length_holder: np array of size [num_batch]
  outputs: 
    logits: np 3-d array of size [num_batch, max_length, num_targets]
  '''
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  layer_in = feats_holder
  count_layer = 1
  for line in nnet_proto:
    line = line.strip()
    if line == '</NnetProto>':
      break
    with tf.variable_scope('layer'+str(count_layer), reuse=reuse):
      layer_out = build_layer(line, layer_in, seq_length = seq_length_holder, 
                              keep_in_prob = keep_in_prob_holder,
                              keep_out_prob = keep_out_prob_holder,
                              reuse = reuse)
      layer_in = layer_out
      count_layer += 1
  logits = layer_out
  return logits


def build_layer(line, layer_in, seq_length = None, keep_in_prob = None, 
                keep_out_prob = None, reuse = False):
  layer_type, info = line.split(' ', 1)
  if layer_type == '<AffineTransform>':
    layer_out = layer.affine_transform(info, layer_in)
  if layer_type == '<LinearTransform>':
    layer_out = layer.linear_transform(info, layer_in)
  elif layer_type == '<BatchNormalization>':
    layer_out = layer.batch_normalization(info, layer_in)
  elif layer_type == '<Sigmoid>':
    layer_out = tf.sigmoid(layer_in)
  elif layer_type == '<Relu>':
    layer_out = tf.nn.relu(layer_in)
  elif layer_type == '<Tanh>':
    layer_out = tf.tanh(layer_in)
  elif layer_type == '<Softmax>':
    layer_out = tf.nn.softmax(layer_in)
  elif layer_type == '<Dropout>':
    layer_out = tf.nn.dropout(layer_in, float(info))
  elif layer_type == '<LSTM>':
    layer_out = layer.lstm(info, layer_in, seq_length, keep_in_prob, keep_out_prob, reuse = reuse)
  elif layer_type == '<BLSTM>':
    layer_out = layer.blstm(info, layer_in, seq_length, keep_in_prob, keep_out_prob, reuse = reuse)

  return layer_out


def loss_dnn(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy-mean')
  return loss


def loss_lstm(logits, labels, mask):
  '''
  args:
    logits: tf tensor of size [batch_size, max_length, num_targets]
    labels: tf tensor of size [batch_size, max_length]
    mask: tf tensor of size [batch_size, max_length]
  '''
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  masked_cross_entropy = tf.multiply(cross_entropy, mask)
  num_elements = tf.reduce_prod(mask.get_shape())
  num_counts = tf.reduce_sum(mask)
  loss = tf.reduce_mean(masked_cross_entropy, name='xentropy-mean') / num_counts * tf.to_float(num_elements)
  return loss


def training(op_conf, loss, learning_rate_holder, scopes = None):
  ''' learning_rate is a place holder
  loss is output of logits
  '''
  if op_conf['op_type'] in ['SGD', 'sgd']:
    op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_holder)
  elif op_conf['op_type'] == 'momentum':
    op = tf.train.MomentumOptimizer(learning_rate = learning_rate_holder, momentum = op_conf['momentum'])
  elif op_conf['op_type'] in ['adagrad', 'Adagrad']:
    op = tf.train.AdagradOptimizer(learning_rate = learning_rate_holder)
  elif op_conf['op_type'] in ['adam', 'Adam']:
    op = tf.train.AdamOptimizer(learning_rate = learning_rate_holder)
  if scopes == None:
    train_op = op.minimize(loss)
  else:
    train_vars = []
    for scope in scopes:
      train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    train_op = op.minimize(loss, var_list = train_vars)
  return train_op


def evaluation_dnn(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def evaluation_lstm(logits, labels, mask):
  '''
  args:
    logits: tf tensor of size [batch_size, max_length, num_targets]
    labels: tf tensor of size [batch_size, max_length]
    mask: tf tensor of size [batch_size, max_length]
  '''
  feat_dim = logits.get_shape()[2]
  logits_reshape = tf.reshape(logits, [-1, int(feat_dim)])
  labels_reshape = tf.reshape(labels, [-1])
  mask_reshape = tf.reshape(mask, [-1])
  correct = tf.nn.in_top_k(logits_reshape, labels_reshape, 1)
  correct = tf.multiply(tf.to_float(correct), mask_reshape)
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is None:
        print("here") 
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
