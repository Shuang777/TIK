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
train=data/train_100k_nodup
train=data/train_10ks
train_ali=exp/tri4_ali_nodup
#train_ali=exp/tri4_ali_10ks

cv=data/train_dev
cv_ali=exp/tri4_ali_dev

lang=data/lang_sw1_tg
gmm=exp/tri4
#exp=exp/tfdnn_5a_10ks_sigmoid_2048x6
exp=exp/tfdnn_5a_kaldi

test=data/eval2000

config=config/swbd.cfg

if $debug; then
  train=data/train_debug
  train_ali=exp/tri4_ali_30kshort
  cv=data/train_dev_debug
  cv_ali=exp/tri4_ali_dev

  exp=exp/tfdnn_5a_debug
fi


$debug && $pdb && debug_args='-m pdb'

if [ $stage -le 0 ]; then
## Train
python3 $debug_args steps_tf/run_tf.py $config $cv $cv_ali $train $train_ali $gmm $exp

$single && exit
fi

x="--use-gpu true --tc-args '-tc 4'"
## Decode
if [ $stage -le 1 ]; then
steps_tf/decode.sh --use-gpu true --tc-args '-tc 4' --nj $nj --cmd "$decode_cmd" \
  $test $gmm/graph_sw1_tg $exp/decode_eval2000
fi

#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali

}
