#!/bin/bash
{

set -e
set -o pipefail

nj=10
stage=0
single=true
debug=false
pdb=true
gpu=1
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
exp=exp/tfdnn_5a_sigmoid_2048x6
#exp=exp/tfdnn_5a_10ks_sigmoid_2048x6_drop0.5
#exp=exp/tfdnn_5a_kaldi
config=config/swbd.cfg

if [ $gpu -gt 1 ]; then
  exp=${exp}_gpu$gpu
  config=config/swbd_multi.cfg
fi

if $debug; then
  train=data/train_debug
  train_ali=exp/tri4_ali_debug
  config=config/swbd_multi.cfg

  exp=exp/tfdnn_5a_debug_multi
fi


$debug && $pdb && debug_args='-m pdb'

if [ $stage -le 0 ]; then
## Train
python3 $debug_args steps_tf/run_tf.py $config $train $train_ali $exp

$single && exit
fi

name=eval2000
## Decode
if [ $stage -le 1 ]; then
steps_tf/decode.sh --nj $nj --cmd "$decode_cmd" --model-name iter01.model.txt\
  --transform-dir exp/tri4/decode_${name}_sw1_tg \
  data/$name $gmm/graph_sw1_tg $exp/decode_$name
fi

#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali

}
