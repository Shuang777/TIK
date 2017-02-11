import tensorflow as tf

def placeholder_inputs(input_dim, batch_size):
  feats_holder = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
  labels_holder = tf.placeholder(tf.int32, shape=(batch_size))
  return feats_holder, labels_holder


def inference(feats_holder, input_dim, hidden1_units, hidden2_units, output_dim):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([input_dim, hidden1_units], stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(feats_holder, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units], stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, output_dim], stddev=0.1),
        name='weights')
    biases = tf.Variable(tf.zeros([output_dim]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def loss(logits, labels):

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy-mean')
  return loss


def training(loss, learning_rate):

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):

  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))
