import tensorflow as tf

def placeholder_inputs(input_dim, batch_size):
  feats_holder = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
  labels_holder = tf.placeholder(tf.int32, shape=(batch_size))
  return feats_holder, labels_holder


def inference(feats_holder, input_dim, hidden_units, num_hidden_layers, output_dim, nonlin = 'relu'):
  layer_in = feats_holder
  for i in xrange(num_hidden_layers):
    with tf.name_scope('hidden'+str(i+1)):
      dim_in = input_dim if i == 0 else hidden_units
      dim_out = hidden_units
      weights = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1), name='weights')
      biases = tf.Variable(tf.zeros([dim_out]), name='biases')
      if nonlin == 'relu':
        layer_out = tf.nn.relu(tf.matmul(layer_in, weights) + biases)
      elif nonlin == 'sigmoid':
        layer_out = tf.sigmoid(tf.matmul(layer_in, weights) + biases)
      layer_in = layer_out
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden_units, output_dim], stddev=0.1), name='weights')
    biases = tf.Variable(tf.zeros([output_dim]), name='biases')
    logits = tf.matmul(layer_in, weights) + biases
  return logits


def loss(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy-mean')
  return loss


def training(loss, learning_rate):
  ''' learning_rate is a place holder
  loss is output of logits
  '''

  optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
