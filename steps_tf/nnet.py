import tensorflow as tf
import math
import layer

def placeholder_inputs(input_dim, batch_size):
  feats_holder = tf.placeholder(tf.float32, shape=(batch_size, input_dim), name='feature')
  labels_holder = tf.placeholder(tf.int32, shape=(batch_size), name='target')
  return feats_holder, labels_holder


def inference(feats_holder, nnet_proto_file):
  # Small epsilon value for the batch normalization transform
  epsilon = 1e-3
  
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


def build_layer(line, layer_in):
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


def loss(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy-mean')
  return loss


def training(op_conf, loss, learning_rate_holder):
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
  train_op = op.minimize(loss)
  return train_op


def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
