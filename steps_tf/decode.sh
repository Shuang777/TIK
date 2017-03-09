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

utt_mode=false

# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: decode.sh [options] <data-dir> <graph-dir> <decode-dir>"
   echo " e.g.: decode.sh data/test exp/tri4/graph exp/dnn_5a/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
graphdir=$2
dir=$3
[ -z $srcdir ] && srcdir=`dirname $dir`;
sdata=$data/split$nj;

mkdir -p $dir/log

if $utt_mode; then
  sdata=$data/split${nj}utt
fi

$utt_mode && utt_opts='--per-utt'

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $utt_opts $data $nj

echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

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
cmds="$cmds python3 steps_tf/nnet_forward.py $srcdir/config $srcdir/final.model.txt $srcdir/ali_train_pdf.counts |"

$cmd $tc_args JOB=1:$nj $dir/log/decode.JOB.log \
  $cmds latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $srcdir/final.mdl $graphdir/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz"
fi

if ! $skip_scoring ; then
  if [ -x mylocal/score.sh ]; then
    mylocal/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
  elif [ -x local/score.sh ]; then
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
  else
    echo "Not scoring because neither mylocal/score.sh nor local/score.sh exists"
    exit 1
  fi
fi

exit 0;
}
