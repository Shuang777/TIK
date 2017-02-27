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


def make_nnet_proto(feat_dim, num_leaves, conf, nnet_proto_file):

  # Optionaly scale
  def Glorot(dim1, dim2):
    if with_glorot:
      # 35.0 = magic number, gives ~1.0 in inner layers for hid-dim 1024dim,
      return 35.0 * math.sqrt(2.0/(dim1+dim2));
    else:
      return 1.0


  nnet_proto = open(nnet_proto_file, 'w')
  num_hid_layers = conf['num_hidden_layers']
  num_hid_neurons = conf['hidden_units']

  # Check
  assert(feat_dim > 0)
  assert(num_leaves > 0)
  assert(num_hid_layers >= 0)
  assert(num_hid_neurons > 0)

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

  nnet_proto.write("<NnetProto>\n")

  if num_hid_layers == 0 and bottleneck_dim != 0:
    # No hidden layer while adding bottleneck means:
    # - add bottleneck layer + hidden layer + output layer
    if bottleneck_trick:
      # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
      nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f\n" % \
       (feat_dim, bottleneck_dim, \
        (param_stddev_factor * Glorot(feat_dim, bottleneck_dim) * 0.75 ), 0.1))
      # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
      nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f\n" % \
       (bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
        (param_stddev_factor * Glorot(bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1))
    else:
      nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f\n" % \
       (feat_dim, bottleneck_dim, \
        (param_stddev_factor * Glorot(feat_dim, bottleneck_dim))))
      nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
       (bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
        (param_stddev_factor * Glorot(bottleneck_dim, num_hid_neurons))))
    nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons)) # Non-linearity
    # Last AffineTransform (10x smaller learning rate on bias)
    nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f\n" % \
     (num_hid_neurons, num_leaves, 0.0, 0.0, \
      (param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1))
    # Optionaly append softmax
    if with_softmax:
        nnet_proto.write("<Softmax> <InputDim> %d <OutputDim> %d\n" % (num_leaves, num_leaves))

  elif num_hid_layers == 0:
    # NO HIDDEN LAYERS!
    # Add only last layer (logistic regression)
    nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
        (feat_dim, num_leaves, 0.0, 0.0, (param_stddev_factor * Glorot(feat_dim, num_leaves))))
    if with_softmax:
      nnet_proto.write("<Softmax> <InputDim> %d <OutputDim> %d\n" % (num_leaves, num_leaves))

  else:
    # THE USUAL DNN PROTOTYPE STARTS HERE!
    # Assuming we have >0 hidden layers,
    assert(num_hid_layers > 0)

    # Begin the prototype,
    # First AffineTranform,
    nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
      (feat_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
       (param_stddev_factor * Glorot(feat_dim, num_hid_neurons) * \
        (math.sqrt(1.0/12.0) if smaller_input_weights else 1.0))))
      # Note.: compensating dynamic range mismatch between input features and Sigmoid-hidden layers,
      # i.e. mapping the std-dev of N(0,1) (input features) to std-dev of U[0,1] (sigmoid-outputs).
      # This is done by multiplying with stddev(U[0,1]) = sqrt(1/12).
      # The stddev of weights is consequently reduced with scale 0.29,
    nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
    if conf.get('dropout', 1.0) != 1.0:
      nnet_proto.write("<Dropout> <InputDim> %d <OutputDim> %d %f\n" % (num_hid_neurons, num_hid_neurons, conf['dropout']))

    # Internal AffineTransforms,
    for i in range(num_hid_layers-1):
      nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
            (num_hid_neurons, num_hid_neurons, hid_bias_mean, hid_bias_range, \
             (param_stddev_factor * Glorot(num_hid_neurons, num_hid_neurons))))
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
      if conf.get('dropout', 1.0) != 1.0:
        nnet_proto.write("<Dropout> <InputDim> %d <OutputDim> %d %f\n" % (num_hid_neurons, num_hid_neurons, conf['dropout']))

    # Optionaly add bottleneck,
    if bottleneck_dim != 0:
      assert(bottleneck_dim > 0)
      if bottleneck_trick:
        # 25% smaller stddev -> small bottleneck range, 10x smaller learning rate
        nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f <LearnRateCoef> %f\n" % \
         (num_hid_neurons, bottleneck_dim, \
          (param_stddev_factor * Glorot(num_hid_neurons, bottleneck_dim) * 0.75 ), 0.1))
        # 25% smaller stddev -> smaller gradient in prev. layer, 10x smaller learning rate for weigts & biases
        nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f\n" % \
         (bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
          (param_stddev_factor * Glorot(bottleneck_dim, num_hid_neurons) * 0.75 ), 0.1, 0.1))
      else:
        # Same learninig-rate and stddev-formula everywhere,
        nnet_proto.write("<LinearTransform> <InputDim> %d <OutputDim> %d <ParamStddev> %f\n" % \
         (num_hid_neurons, bottleneck_dim, \
          (param_stddev_factor * Glorot(num_hid_neurons, bottleneck_dim))))
        nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f\n" % \
         (bottleneck_dim, num_hid_neurons, hid_bias_mean, hid_bias_range, \
          (param_stddev_factor * Glorot(o.bottleneck_dim, num_hid_neurons))))
      nnet_proto.write("<%s> <InputDim> %d <OutputDim> %d\n" % (conf['nonlin'], num_hid_neurons, num_hid_neurons))
      if conf.get('dropout', 1.0) != 1.0:
        nnet_proto.write("<Dropout> <InputDim> %d <OutputDim> %d %s\n" % (num_hid_neurons, num_hid_neurons, conf['dropout']))

    # Last AffineTransform (10x smaller learning rate on bias)
    nnet_proto.write("<AffineTransform> <InputDim> %d <OutputDim> %d <BiasMean> %f <BiasRange> %f <ParamStddev> %f <LearnRateCoef> %f <BiasLearnRateCoef> %f\n" % \
      (num_hid_neurons, num_leaves, 0.0, 0.0, \
       (param_stddev_factor * Glorot(num_hid_neurons, num_leaves)), 1.0, 0.1))

    # Optionaly append softmax
    if with_softmax:
      nnet_proto.write("<Softmax> <InputDim> %d <OutputDim> %d\n" % (num_leaves, num_leaves))

  nnet_proto.write("</NnetProto>\n")
  nnet_proto.close()

