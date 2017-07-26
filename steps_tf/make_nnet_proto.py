# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
#                2017  International Computer Science Institute (author: Hang Su)      

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Generated Nnet prototype

import math

def Glorot(dim1, dim2, with_glorot):
  if with_glorot:
    # 35.0 = magic number, gives ~1.0 in inner layers for hid-dim 1024dim,
    return 35.0 * math.sqrt(2.0/(dim1+dim2));
  else:
    return 1.0


def make_lstm_proto(feat_dim, output_dim, conf, nnet_proto_file):
  nnet_proto = open(nnet_proto_file, 'w')
  num_hid_layers = conf['num_hidden_layers']
  num_cells = conf['num_cells']
  lstm_type = conf['lstm_type']
  with_glorot = conf.get('with_glorot', True)
  
  #Use batch normalization for affine transform
  batch_norm = conf.get('batch_norm', False)
  
  #Factor to rescale Normal distriburtion for initalizing weight matrices
  param_stddev_factor = conf.get('param_stddev_factor', 0.1)

  if batch_norm:
    affine_layer = 'BatchNormalization'
  else:
    affine_layer = 'AffineTransform'

  nnet_proto.write("<NnetProto>\n")

  for i in range(num_hid_layers):
    nnet_proto.write("<%s> <NumCells> %d <KeepInProb> %f <KeepOutProb> %f\n" % (lstm_type, num_cells, conf.get('keep_in_prob', 1.0), conf.get('keep_out_prob', 1.0)))
  
  nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f\n" % \
      (affine_layer, num_cells, output_dim, 0.0, 0.0, \
       (param_stddev_factor * Glorot(num_cells, output_dim, with_glorot)), 1.0, 0.1))

  nnet_proto.write("</NnetProto>\n")
  nnet_proto.close()


def make_nnet_proto(feat_dim, output_dim, conf, nnet_proto_file):

  nnet_proto = open(nnet_proto_file, 'w')
  num_hid_layers = conf['num_hidden_layers']
  num_hid_neurons = conf['hidden_units']
  num_hid_layers_after_bn = conf.get('num_hidden_layers_after_bn', 1)
  
  with_glorot = conf.get('with_glorot', True)

  # Check
  assert(feat_dim > 0)
  assert(output_dim > 0)
  assert(num_hid_layers >= 0)
  assert(num_hid_neurons > 0)

  #Use batch normalization for affine transform
  batch_norm = conf.get('batch_norm', False)

  if batch_norm:
    affine_layer = 'BatchNormalization'
  else:
    affine_layer = 'AffineTransform'
  hid_bias_mean = conf.get('hid_bias_mean', -2.0)

  #Set bias range for hidden activations (+/- 1/2 range around mean)
  hid_bias_range = conf.get('hid_bias_range', 4.0)

  #Factor to rescale Normal distriburtion for initalizing weight matrices
  param_stddev_factor = conf.get('param_stddev_factor', 0.1)

  #Generate normalized weights according to X.Glorot paper, but mapping U->N with same variance (factor sqrt(x/(dim_in+dim_out)))'
  with_glorot = conf.get('with_glorot', True)

  #1/12 reduction of stddef in input layer [default: %default]
  smaller_input_weights = conf.get('smaller_input_weights', False)

  #Smaller initial weights and learning rate around bottleneck
  bottleneck_trick = conf.get('bottleneck_trick', True)

  #Make bottleneck network with desired bn-dim (0 = no bottleneck)
  bottleneck_dim = conf.get('bottleneck_dim', 0)

  #Add softmax layer at the end
  with_softmax = conf.get('with_softmax', True)

  #Use batch normalization for affine transform
  batch_norm = conf.get('batch_norm', False)

  if batch_norm:
    affine_layer = 'BatchNormalization'
  else:
    affine_layer = 'AffineTransform'

  nnet_proto.write("<NnetProto>\n")

  # Begin the prototype,
  # First AffineTranform,
  nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
    (affine_layer, feat_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
     (param_stddev_factor * Glorot(feat_dim, num_hid_neurons, with_glorot) * \
      (math.sqrt(1.0/12.0) if smaller_input_weights else 1.0))))
    # Note.: compensating dynamic range mismatch between input features and Sigmoid-hidden layers,
    # i.e. mapping the std-dev of N(0,1) (input features) to std-dev of U[0,1] (sigmoid-outputs).
    # This is done by multiplying with stddev(U[0,1]) = sqrt(1/12).
    # The stddev of weights is consequently reduced with scale 0.29,
  nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
  if conf.get('keep_prob', 1.0) != 1.0:
    nnet_proto.write("<Dropout> %f\n" % conf['keep_prob'])

  # Internal AffineTransforms,
  for i in range(num_hid_layers-1):
    nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
          (affine_layer, num_hid_neurons, num_hid_neurons, hid_bias_mean, hid_bias_range, \
           (param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons, with_glorot))))
    nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
    if conf.get('keep_prob', 1.0) != 1.0:
      nnet_proto.write("<Dropout> %f\n" % conf['keep_prob'])

  # Optionaly add bottleneck,
  if bottleneck_dim != 0:
    assert(bottleneck_dim > 0)
    nnet_proto.write("<BottleNeck>\n")
    if bottleneck_trick:
      # 25% smaller stddev -> small bottleneck range
      nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f\n" % \
       (num_hid_neurons, bottleneck_dim, \
        (param_stddev_factor * Glorot(num_hid_neurons, bottleneck_dim, with_glorot) * 0.75 )))
      nnet_proto.write("</BottleNeck>\n")
      # 25% smaller stddev -> smaller gradient in prev. layer
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
       (affine_layer, bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
        (param_stddev_factor * Glorot(bottleneck_dim, num_hid_neurons, with_glorot) * 0.75 )))
    else:
      # Same learninig-rate and stddev-formula everywhere,
      nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f\n" % \
       (num_hid_neurons, bottleneck_dim, \
        (param_stddev_factor * Glorot(num_hid_neurons, bottleneck_dim, with_glorot))))
      nnet_proto.write("</BottleNeck>\n")
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
       (affine_layer, bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
        (param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons, with_glorot))))
    nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
    if conf.get('keep_prob', 1.0) != 1.0:
      nnet_proto.write("<Dropout> %s\n" % conf['keep_prob'])

    for i in range(num_hid_layers_after_bn-1):
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
            (affine_layer, num_hid_neurons, num_hid_neurons, hid_bias_mean, hid_bias_range, \
             (param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons, with_glorot))))
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
      if conf.get('keep_prob', 1.0) != 1.0:
        nnet_proto.write("<Dropout> %f\n" % conf['keep_prob'])

  # Last AffineTransform (10x smaller learning rate on bias)
  nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
    (affine_layer, num_hid_neurons, output_dim, 0.0, 0.0, \
     (param_stddev_factor * Glorot(num_hid_neurons, output_dim, with_glorot))))

  # Optionaly append softmax
  if with_softmax:
    nnet_proto.write("<Softmax> <InputDim> %d <OutputDim> %d\n" % (output_dim, output_dim))

  nnet_proto.write("</NnetProto>\n")
  nnet_proto.close()

