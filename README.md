# TiwK
Tensorflow integration with Kaldi

Supports frame-level DNN hybrid model, LSTM model

# TODO
a separate feat\_holder and outputs / logits for decoding lstm model or joint dnn model; 
currently we are using the same feat\_holder as in training and it wastes time when we do decoding.
