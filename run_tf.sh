#!/bin/bash
{

set -e
set -o pipefail

nj=4
stage=0
single=true
debug=false
. ./cmd.sh
. ./path.sh
. parse_options.sh

## Configurable directories
#train=data/train_nodup
train=data/train_100k_nodup
train_ali=exp/tri4_ali_nodup
train=data/train_debug
train_ali=exp/tri4_ali_30kshort

if $debug; then
  train=data/train_debug
  train_ali=exp/tri4_ali_30kshort
fi

cv=data/train_dev_debug
cv_ali=exp/tri4_ali_dev

lang=data/lang_sw1_tg
gmm=exp/tri4
exp=exp/tfdnn_5a


$debug && debug_args='-m pdb'
if [ $stage -le 0 ]; then
## Train
python $debug_args steps_tf/run_tf.py $cv $cv_ali $train $train_ali $gmm $exp
exit

## Get priors: Make a Python script to do this.
ali-to-pdf $gmm/final.mdl ark:"gunzip -c ${gmm}_ali/ali.*.gz |" ark,t:- | \
    cut -d" " -f2- | tr ' ' '\n' | sed -r '/^\s*$/d' | sort | uniq -c | sort -n -k2 | \
    awk '{a[$2]=$1; c+=$1; LI=$2} END{for(i=0;i<LI;i++) printf "%e,",a[i]/c; printf "%e",a[LI]/c}' \
    > $exp/dnn.priors.csv

$single && exit

fi

graph=$gmm/graph


## Decode
teps_kt/decode.sh --nj $nj \
    --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
    $test $gmm/graph $exp $exp/decode

#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali

}
