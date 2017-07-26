#!/bin/bash
{
set -e
set -o pipefail

# Being configuration
stage=0
nj=10
tc_args=
cmd=run.pl

srcdir=
transform_dir=

max_active=7000 # max-active
beam=13.0 # beam used
latbeam=8.0 # beam used in getting lattices
acwt=0.08333 # acoustic weight used in getting lattices
scoring_opts=
skip_scoring=false

splice_opts=
norm_vars=
add_deltas=

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

[ -z $srcdir ] && srcdir=`dirname $dir`;
sdata=$data/split$nj;

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $utt_opts $data $nj

if [ $stage -le 0 ]; then
## Set up the features
  cmds="apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
  cmds="$cmds splice-feats $splice_opts ark:- ark:- |"
  cmds="$cmds transform-feats $srcdir/final.mat ark:- ark:- |"
  if [ ! -z "$transform_dir" ]; then
    nj_orig=$(cat $transform_dir/num_jobs)
    if [ $nj_orig != $nj ]; then
      for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
        copy-feats ark:- ark,scp:$dir/trans.ark,$dir/trans.scp
      cmds="$cmds transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.ark ark:- ark:- |"
    else
      cmds="$cmds transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
    fi
  fi
  
  utils/copy_data_dir.sh $data $tgtdir

  $cmd $tc_args JOB=1:$nj $dir/log/gen_bn.JOB.log \
    $cmds python3 steps_tf/nnet_forward.py $srcdir/config $srcdir/$model_name \| \
    copy-feats ark:- ark,scp:`pwd`/$dir/bn_feats.JOB.ark,$dir/bn_feats.JOB.scp

  for i in `seq $nj`; do
    cat $dir/bn_feats.$i.scp
  done > $tgtdir/feats.scp

  steps/compute_cmvn_stats.sh $tgtdir
fi

exit 0;
}
