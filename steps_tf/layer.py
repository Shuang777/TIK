import tensorflow as tf

def info2dict(info):
  ''' parse a string like "<key-1> field1 <key-2> field2" to a dictionary'''
  span = 2
  fields = info.split()
  return dict([(fields[i], fields[i+1]) for i in range(0, len(fields), span)])


def affine_transform(info, layer_in):
  info_dict = info2dict(info)
  
  input_dim = int(info_dict['<InputDim>'])
  output_dim = int(info_dict['<OutputDim>'])
  stddev = float(info_dict['<ParamStddev>'])
  minval = float(info_dict['<BiasMean>']) - float(info_dict['<BiasRange>'])/2
  maxval = float(info_dict['<BiasMean>']) + float(info_dict['<BiasRange>'])/2

  weights = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                          stddev = stddev),
                        name = 'weights')
  biases = tf.Variable(tf.random_uniform([output_dim], 
                         minval = minval, 
                         maxval = maxval),
                       name = 'biases')

  if len(layer_in.get_shape()) == 2:
    layer_out = tf.matmul(layer_in, weights) + biases
  elif len(layer_in.get_shape()) == 3:   # this is after a LSTM layer, we need to do some reshaping
    batch_size, max_length = layer_in.get_shape()[:2]
    layer_out = tf.reshape(layer_in, [-1, input_dim])
    layer_out = tf.matmul(layer_out, weights) + biases
    layer_out = tf.reshape(layer_out, [int(batch_size), -1, output_dim])

  return layer_out


#def linear_transform(info):


def batch_normalization(info, layer_in):
  # Small epsilon value for the batch normalization transform
  epsilon = 1e-3

  info_dict = info2dict(info)
  
  input_dim = int(info_dict['<InputDim>'])
  output_dim = int(info_dict['<OutputDim>'])
  stddev = float(info_dict['<ParamStddev>'])

  weights = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                          stddev = stddev),
                        name = 'weights')
  
  z = tf.matmul(layer_in, weights)
  batch_mean, batch_var = tf.nn.moments(z, [0])
  scale = tf.Variable(tf.ones([output_dim]), 'scale')
  beta = tf.Variable(tf.zeros([output_dim]), 'beta')
  layer_out = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, epsilon)

  return layer_out


def lstm(info, layer_in, seq_length):
  info_dict = info2dict(info)
  
  num_cell = int(info_dict['<NumCells>'])
  num_layers = int(info_dict['<NumLayers>'])
  keep_prob = float(info_dict['<KeepProb>'])

  cell = tf.nn.rnn_cell.LSTMCell(num_cell, state_is_tuple=True)

  cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * num_layers, state_is_tuple=True)

  layer_out,_ = tf.nn.dynamic_rnn(cell, 
                  layer_in,
                  sequence_length = seq_length,
                  dtype=tf.float32)

  
  return layer_out
