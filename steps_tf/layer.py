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

  truncated_normal_initializer = tf.truncated_normal_initializer(mean = 0, stddev = stddev)
  random_uniform_initializer = tf.random_uniform_initializer(minval = minval, maxval = maxval)

  weights = tf.get_variable(name = 'weights', shape = [input_dim, output_dim],
                            initializer = truncated_normal_initializer)
  biases = tf.get_variable(name = 'biases', shape = [output_dim], 
                           initializer = random_uniform_initializer)
  
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, biases)

  if len(layer_in.get_shape()) == 2:
    layer_out = tf.matmul(layer_in, weights) + biases
  elif len(layer_in.get_shape()) == 3:   # this is of size [num_batch, num_frame, feat_dim]
    batch_size, max_length = layer_in.get_shape()[:2]
    layer_out = tf.reshape(layer_in, [-1, input_dim])
    layer_out = tf.matmul(layer_out, weights) + biases
    layer_out = tf.reshape(layer_out, [int(batch_size), -1, output_dim])
  else:
    raise RuntimeError("affine_transform: does not support layer_in of shape %s" % layer_in.get_shape())

  return layer_out


def linear_transform(info, layer_in):
  info_dict = info2dict(info)
  
  input_dim = int(info_dict['<InputDim>'])
  output_dim = int(info_dict['<OutputDim>'])
  stddev = float(info_dict['<ParamStddev>'])

  truncated_normal_initializer = tf.truncated_normal_initializer(mean = 0, stddev = stddev)
  weights = tf.get_variable(name = 'weights', shape = [input_dim, output_dim],
                            initializer = truncated_normal_initializer)
  
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

  layer_out = tf.matmul(layer_in, weights)

  return layer_out


def affine_batch_normalization(info, layer_in):
  # Small epsilon value for the batch normalization transform
  epsilon = 1e-3

  info_dict = info2dict(info)
  
  input_dim = int(info_dict['<InputDim>'])
  output_dim = int(info_dict['<OutputDim>'])
  stddev = float(info_dict['<ParamStddev>'])

  truncated_normal_initializer = tf.truncated_normal_initializer(mean = 0, stddev = stddev)
  weights = tf.get_variable(name = 'weights', shape = [input_dim, output_dim],
                            initializer = truncated_normal_initializer)

  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
  
  if len(layer_in.get_shape()) == 2:
    z = tf.matmul(layer_in, weights)
  elif len(layer_in.get_shape()) == 3:   # this is of size [num_batch, num_frame, feat_dim]
    batch_size, max_length = layer_in.get_shape()[:2]
    z = tf.reshape(layer_in, [-1, input_dim])
    z = tf.matmul(z, weights)
    z = tf.reshape(z, [int(batch_size), -1, output_dim])

  assert(len(layer_in.get_shape()) in [2, 3])
  axes = [0] if len(layer_in.get_shape()) == 2 else [0, 1]
  batch_mean, batch_var = tf.nn.moments(z, axes)
  scale = tf.get_variable(name = 'scale', shape = [output_dim], initializer = tf.ones_initializer())
  beta = tf.get_variable(name = 'beta', shape = [output_dim], initializer = tf.zeros_initializer())
  layer_out = tf.nn.batch_normalization(z, batch_mean, batch_var, beta, scale, epsilon)

  return layer_out


def batch_normalization(info, layer_in):
  epsilon = 1e-3
  info_dict = info2dict(info)
  input_dim = int(info_dict['<InputDim>'])
  output_dim = int(info_dict['<OutputDim>'])

  assert(len(layer_in.get_shape()) in [2, 3])
  axes = [0] if len(layer_in.get_shape()) == 2 else [0, 1]
  batch_mean, batch_var = tf.nn.moments(layer_in, axes)
  scale = tf.get_variable(name = 'scale', shape = [output_dim], initializer = tf.ones_initializer())
  beta = tf.get_variable(name = 'beta', shape = [output_dim], initializer = tf.zeros_initializer())
  layer_out = tf.nn.batch_normalization(layer_in, batch_mean, batch_var, beta, scale, epsilon)
  return layer_out


def lstm(info, layer_in, seq_length, keep_in_prob, keep_out_prob, reuse = False):
  info_dict = info2dict(info)
  
  num_cell = int(info_dict['<NumCells>'])
  use_peepholes = info_dict.get('<UsePeepHoles>', 'False').lower() == 'true'
  if '<NumProj>' in info_dict:
    num_proj = int(info_dict['<NumProj>'])
    cell = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                   use_peepholes = use_peepholes, num_proj = num_proj)
  else:
    cell = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                   use_peepholes = use_peepholes)

  cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = keep_in_prob, 
                                       output_keep_prob = keep_out_prob)

  layer_out,_ = tf.nn.dynamic_rnn(cell, 
                  layer_in,
                  sequence_length = seq_length,
                  dtype=tf.float32)

  
  return layer_out


def blstm(info, layer_in, seq_length, keep_in_prob, keep_out_prob, reuse = False):
  info_dict = info2dict(info)
  
  num_cell = int(info_dict['<NumCells>']) / 2
  use_peepholes = info_dict.get('<UsePeepHoles>', 'False').lower() == 'true'

  if '<NumProj>' in info_dict:
    num_proj = int(info_dict['<NumProj>']) / 2
    cell_fw = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                      use_peepholes = use_peepholes, num_proj = num_proj)
    cell_bw = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                      use_peepholes = use_peepholes, num_proj = num_proj)
  else:
    cell_fw = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                      use_peepholes = use_peepholes)
    cell_bw = tf.contrib.rnn.LSTMCell(num_cell, state_is_tuple = True, reuse = reuse,
                                      use_peepholes = use_peepholes)

  cell_fw = tf.contrib.rnn.DropoutWrapper(cell = cell_fw, input_keep_prob = keep_in_prob, 
                                          output_keep_prob = keep_out_prob)
  cell_bw = tf.contrib.rnn.DropoutWrapper(cell = cell_bw, input_keep_prob = keep_in_prob, 
                                          output_keep_prob = keep_out_prob)

  rnn_outs,_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                  cell_bw,
                  layer_in,
                  sequence_length = seq_length,
                  dtype=tf.float32)

  layer_out = tf.concat(rnn_outs, 2)
  return layer_out


def pooling(info, layer_in, mask, reuse = False):
  '''
  layer_in: 3-d np array of size [num_batch, max_length, hidden_dim]
  mask: 2-d np array of size [num_batch, max_length]
  '''
  info_dict = info2dict(info)
  mask_shape = mask.get_shape().as_list()
  mask_reshape = tf.reshape(mask, [mask_shape[0], mask_shape[1], 1])
  num_batch, max_length, dim_hid = layer_in.get_shape().as_list()
  mask_tile = tf.tile(mask_reshape, [1, 1, dim_hid])
  masked = tf.multiply(mask_tile, layer_in)
  mean, var = tf.nn.moments(masked, [1])

  mean_trans = tf.transpose(tf.scalar_mul(max_length, mean))
  scales = tf.reduce_sum(mask, 1)
  stats = tf.divide(mean_trans, scales)

  use_std = info_dict.get('<UseStd>', 'True').lower() == 'true'
  if use_std:
    var_trans = tf.transpose(tf.scalar_mul(max_length, var))
    var_scaled = tf.divide(var_trans, scales)
    stats = tf.concat([stats, var_scaled], 0)

  layer_out = tf.transpose(stats)

  return layer_out
