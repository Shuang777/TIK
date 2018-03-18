#!/bin/bash
{
set -e
set -o pipefail

# Being configuration
stage=0
nj=10
tc_args=
cmd=run.pl

transform_dir=

splice_opts=
norm_vars=
add_deltas=

use_gpu=false
model_name=final.model.txt
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <data-dir> <tgt-dir> <data-dir>"
   echo " e.g.: $0 data/train data/train_bn data/test_bn"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
tgtdir=$2
dir=$3

sdata=$data/split$nj;

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $utt_opts $data $nj

## Set up the features
if [ ! -z "$transform_dir" ]; then
  nj_orig=$(cat $transform_dir/num_jobs)
  if [ $nj_orig -eq $nj ]; then
    trans=trans.JOB
  else
    for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
      copy-feats ark:- ark,scp:$dir/trans.ark,$dir/trans.scp
    trans=trans.ark
  fi
fi

utils/copy_data_dir.sh $data $tgtdir

if $use_gpu; then  gpu_opts="--use-gpu --gpu-id JOB"; fi

if [ $stage -le 0 ]; then
  $cmd $tc_args JOB=1:$nj $dir/log/gen_bn.JOB.log \
    python steps_tf/nnet_forward.py $gpu_opts --no-softmax \
    --transform $transform_dir/$trans $sdata/JOB $dir/$model_name \| \
    copy-feats ark:- ark,scp:`pwd`/$dir/bn_feats.JOB.ark,$dir/bn_feats.JOB.scp

  for i in `seq $nj`; do
    cat $dir/bn_feats.$i.scp
  done > $tgtdir/feats.scp

  steps/compute_cmvn_stats.sh $tgtdir
fi

exit 0;
}
