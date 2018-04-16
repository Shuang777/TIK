import os

def read_int_or_none(file_name):
  if os.path.isfile(file_name):
    return int(open(file_name).read())
  else:
    return None

def parse_int_or_list(file_name):
  output = open(file_name).read()
  if output[0] == '(' or output[0] == '[':
    output = output[1:-1].split(',')
    output_dim = [int(x) for x in output]
  else:
    output_dim = int(output)
  return output_dim

