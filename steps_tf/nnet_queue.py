import numpy as np

class NNSeqQueue(object):
  ''' a queue for accepting features and perform nnet forward task '''

  def __init__(self, nnet, writer, embedding_index = 0):
    self.nnet = nnet
    self.writer = writer
    self.embedding_index = embedding_index

    self.max_length = nnet.get_max_length()
    self.batch_size = nnet.get_batch_size()

    self.reset()


  def pack_feats(self, uid, feats):
    '''
    inputs:
      uid: string
      feats: 2d np-array of size [num_frames, feat_dim]

    pack to 3d np-array of size [batch, max_length, feat_dim]
    and mask 2d np-array of size [batch, max_length]
      
    '''
    feats_packed = []
    mask = []
    start_index = 0
    while start_index + self.max_length <= len(feats):
      end_index = start_index + self.max_length
      feats_packed.append(feats[start_index:end_index])
      mask.append(np.ones(self.max_length))
      start_index += self.max_length

    if start_index + self.max_length > len(feats):
      # last segment, we shift a bit so we have a full segment
      start_index = max(len(feats) - self.max_length, 0)
      num_zeros = start_index + self.max_length - len(feats)
      zeros2pad = np.zeros((num_zeros, len(feats[0])))
      feats_packed.append(np.concatenate((feats[start_index:len(feats)], zeros2pad)))
      mask.append(np.append(np.ones(len(feats)-start_index), np.zeros(num_zeros)))
    
    utt_list = [uid] * len(feats_packed)
    return feats_packed, mask, utt_list


  def add2queue(self, uid, feats):
    feats_packed, mask, utt_list = self.pack_feats(uid, feats)
    # now add these to stack
    self.feats_queue.extend(feats_packed)
    self.mask_queue.extend(mask)
    self.utt_list_queue.extend(utt_list)

    if len(self.feats_queue) >= self.batch_size:
      # now, time to send to nnet
      batch_feats = np.array(self.feats_queue[0:self.batch_size])
      batch_mask = np.array(self.mask_queue[0:self.batch_size])
      batch_utt_list = self.utt_list_queue[0:self.batch_size]
    
      self.process_batch(batch_feats, batch_mask, batch_utt_list)

      self.feats_queue = self.feats_queue[self.batch_size:]
      self.mask_queue = self.mask_queue[self.batch_size:]
      self.utt_list_queue = self.utt_list_queue[self.batch_size:]


  def process_batch(self, batch_feats, batch_mask, batch_utt_list):

    assert(len(batch_feats) == len(batch_mask))
    assert(len(batch_feats) == len(batch_utt_list))

    batch_embeddings = self.nnet.gen_embedding(batch_feats, batch_mask, self.embedding_index)

    # we need a separate cache for embedding processing
    for i in range(len(batch_utt_list)):
      if batch_utt_list[i] != self.last_utt:
        if self.last_utt is not None:
          utt_embedding = self.acc_embedding / self.embedding_count
          self.writer.write(self.last_utt, utt_embedding)
        # a new speaker
        self.last_utt = batch_utt_list[i]
        self.acc_embedding = batch_embeddings[i]
        self.embedding_count = 1
      else:
        self.acc_embedding += batch_embeddings[i]
        self.embedding_count += 1


  def close(self):
    if len(self.feats_queue) != 0:
      num_seg2pad = self.batch_size - len(self.feats_queue)
      
      batch_feats = np.array(self.feats_queue[0:self.batch_size])
      batch_mask = np.array((self.mask_queue[0:self.batch_size]))

      feats_padded = np.zeros((num_seg2pad, self.max_length, len(self.feats_queue[0][0])))
      batch_feats = np.concatenate((batch_feats, feats_padded))
      batch_mask = np.concatenate((batch_mask, np.ones((num_seg2pad, self.max_length))))
      batch_utt_list = self.utt_list_queue[0:self.batch_size] + [None]*num_seg2pad

      self.process_batch(batch_feats, batch_mask, batch_utt_list)

      self.reset()

  
  def reset(self):
    self.feats_queue = []
    self.mask_queue = []
    self.utt_list_queue = []
    
    self.acc_embedding = None
    self.last_utt = None
    self.embedding_count = 0

