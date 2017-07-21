#!/bin/bash
{

set -e
set -o pipefail

nj=10
stage=0
single=true
debug=false
pdb=true

. ./cmd.sh
. ./path.sh
. parse_options.sh

## Configurable directories
train=data/train_nodup
#train=data/train_100k_nodup
train=data/train_10ks
train_ali=exp/tri4_ali_nodup

lang=data/lang_sw1_tg
gmm=exp/tri4
exp=exp/tflstm_5a_10ks_1024x6_conx0_max60_win20_lr0.4_drop0.8_batch64_jitter_re
#exp=exp/tflstm_5a_10ks_1024x4_win20_lr0.4_jitter
#exp=exp/tflstm_5a_256x4_lr0.04_drop0.8
#exp=exp/tflstm_5a_256x4_lr0.01
#exp=exp/tflstm_5a_1024x4_win20_lr0.1_jitter
#exp=exp/tflstm_5a_10ks_1024x6_conx0_max60_win20_lr0.4_drop0.8_batch32_jitter_gpu2
#exp=exp/tflstm_5a_10ks_1024x6_conx0_max60_win20_lr0.4_drop0.8_batch64_jitter_more
#exp=exp/tflstm_5a_10ks_1024x6_conx0_max60_win20_lr0.4_drop0.8_jitter_re_con

config=config/swbd_lstm_jitter.cfg

if $debug; then
  train=data/train_debug
  train_ali=exp/tri4_ali_debug
  config=config/swbd_lstm_jitter.cfg

  exp=exp/tflstm_5a_debug
fi


$debug && $pdb && debug_args='-m pdb'

if [ $stage -le 0 ]; then
## Train
python3 $debug_args steps_tf/run_tf.py $config $train $train_ali $gmm $exp

$single && exit
fi

name=eval2000
## Decode
if [ $stage -le 1 ]; then
steps_tf/decode.sh --nj $nj --cmd "$decode_cmd" \
  --transform-dir exp/tri4/decode_${name}_sw1_tg \
  data/$name $gmm/graph_sw1_tg $exp/decode_${name}

$single && exit
fi

if [ $stage -le 2 ]; then
#  for iter in $(seq -w 10); do
  for iter in 04 06 08 10 12 13 11 05 07 09 03 01; do
    steps_tf/decode.sh --nj $nj --cmd "$decode_cmd" \
      --transform-dir exp/tri4/decode_${name}_sw1_tg \
      --model-name iter${iter}.model.txt \
      data/$name $gmm/graph_sw1_tg $exp/decode_iter${iter}_${name}
  done
fi


#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali

}
