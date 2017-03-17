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


def inference_dnn(feats_holder, nnet_proto_file):
  
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  layer_in = feats_holder
  count_layer = 1
  for line in nnet_proto:
    line = line.strip()
    if line == '</NnetProto>':
      break
    with tf.name_scope('layer'+str(count_layer)):
      layer_out = build_layer(line, layer_in)
      layer_in = layer_out
      count_layer += 1
  logits = layer_out
  return logits


def scan_subnnet(nnet_proto_file):
  nnet_proto = open(nnet_proto_file, 'r')
  count_subnnet = 0
  for line in nnet_proto:
    line = line.strip()
    if line == '<SubNnet>':
      count_subnnet += 1
  return count_subnnet


def inference_multi(feats_holders, switch_holders, nnet_proto_file):
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  line = nnet_proto.readline().strip()
  assert line == '<MultiSwitch>'
  count_subnnet = 0

  # subnnet part
  line = nnet_proto.readline().strip()
  subnnet_outs = []
  while line:
    if line == '<SubNnet>':
      layer_in = feats_holders[count_subnnet]
      line = nnet_proto.readline().strip()
      with tf.name_scope('layer0_sub'+str(count_subnnet)):
        while line:
          layer_out = build_layer(line, layer_in)
          layer_in = layer_out
          line = nnet_proto.readline().strip()
          if line == '</SubNnet>':
            subnnet_outs.append(layer_out)
            break
    if line == '</SubNnet>':
      count_subnnet += 1
    elif line == '</MultiSwitch>':
    #done with subnnet part
      break;
    line = nnet_proto.readline().strip()

  # merge part
  with tf.name_scope('layer_merge'):
    for i in range(count_subnnet):
      if i == 0:
        layer_out = tf.scalar_mul(switch_holders[0], subnnet_outs[0])
      else:
        layer_out = tf.add(layer_out, tf.scalar_mul(switch_holders[i], subnnet_outs[i]))
  
  # shared hidden part
  count_layer = 1
  layer_in = layer_out
  with tf.name_scope('layer_shared'):
    for line in nnet_proto:
      line = line.strip()
      if line == '</NnetProto>':
        break
      with tf.name_scope('layer'+str(count_layer)):
        layer_out = build_layer(line, layer_in)
      layer_in = layer_out
      count_layer += 1
  logits = layer_out
  return logits


def inference_lstm(feats_holder, seq_length_holder, nnet_proto_file):
  
  nnet_proto = open(nnet_proto_file, 'r')
  line = nnet_proto.readline().strip()
  assert line == '<NnetProto>'
  layer_in = feats_holder
  count_layer = 1
  for line in nnet_proto:
    line = line.strip()
    if line == '</NnetProto>':
      break
    with tf.name_scope('layer'+str(count_layer)):
      layer_out = build_layer(line, layer_in, seq_length = seq_length_holder)
      layer_in = layer_out
      count_layer += 1
  logits = layer_out
  return logits


def build_layer(line, layer_in, seq_length = None):
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
    layer_out = layer.lstm(info, layer_in, seq_length)

  return layer_out


def inference_from_file(feats_holder, input_dim, output_dim, init_file):
  layer_in = feats_holder
  nnet = open(init_file, 'r')
  line = nnet.readline()
  assert line.startswith('<Nnet>')
  i = 0
  for line in nnet:
    if line.startswith('</Nnet>'):
      break
    if line.startswith('<AffineTransform>'):
      info = line.split()
      dim_out = int(info[1])
      dim_in = int(info[2])
      line = nnet.readline()
      assert line.startswith('<LearnRateCoef>')
      line = nnet.readline()
      assert line.strip().startswith('[')
      line = nnet.readline().strip()
      
      mat = []
      while not line.startswith('['):
        if line.endswith(']'):
          line = line.strip(']').strip()
        mat.append(list(map(float, line.split())))
        line = nnet.readline().strip()

      w = list(zip(*mat))
      b = list(map(float, line.split()[1:-1]))
      
      line = nnet.readline()
      assert line.startswith('<!EndOfComponent>')

      line = nnet.readline()
      if line.startswith('<Sigmoid>'):
        with tf.name_scope('hidden'+str(i+1)):
          weights = tf.Variable(w, name='weights')
          biases = tf.Variable(b, name='biases')
          layer_out = tf.sigmoid(tf.matmul(layer_in, weights) + biases)
      elif line.startswith('<Softmax>'):
        with tf.name_scope('softmax_linear'):
          weights = tf.Variable(w, name='weights')
          biases = tf.Variable(b, name='biases')
          logits = tf.add(tf.matmul(layer_in, weights), biases, name = 'logits')

      line = nnet.readline()
      assert line.startswith('<!EndOfComponent>')

      layer_in = layer_out
      i += 1

  return logits


def loss_dnn(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_sum(cross_entropy, name='xentropy-mean')
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
      logits, labels, name='xentropy')
  masked_cross_entropy = tf.multiply(cross_entropy, mask)
  loss = tf.reduce_sum(masked_cross_entropy, name='xentropy-mean')
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


def evaluation(logits, labels):

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
