def str2boolean(string):
  if string in ['True', 'true', 'TRUE']:
    return True
  elif string in ['False', 'false', 'FALSE']:
    return False


def parse(config_tuple):
  config_dict = dict(config_tuple)
  config_parsed = {}
  for i in config_dict.keys():
    if i in ['min_iters', 'keep_lr_iters', 'max_iters', 'context_width', 
             'batch_size', 'hidden_units', 'num_hidden_layers']:
      config_parsed[i] = int(config_dict[i])
    elif i in ['halving_factor', 'start_halving_impr', 'end_halving_impr', 
             'initial_learning_rate', 'momentum', 'keep_prob']:
      config_parsed[i] = float(config_dict[i])
    elif i in ['batch_norm', 'with_softmax']:
      config_parsed[i] = str2boolean(config_dict[i])
    elif i in ['init_file', 'nonlin', 'op_type']:
      config_parsed[i] = config_dict[i]
    else:
      raise RuntimeError('section_config.parse: config field %s not supported' % i)

  return config_parsed
