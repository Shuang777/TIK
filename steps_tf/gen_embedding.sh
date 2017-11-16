#!/bin/bash
{
set -e
set -o pipefail

# Being configuration
stage=0
nj=10
tc_args=
utt_mode=false
use_gpu=false
cmd=run.pl
model_name=final.model.txt
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: decode.sh [options] <data-dir> <nnet-dir> <xvector-dir>"
   echo " e.g.: decode.sh data/sre10_test exp/tfsid_debug exp/xvector_test"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
nnetdir=$2
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
for f in $nnetdir/$model_name $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

if $use_gpu; then  gpu_opts='--use-gpu'; fi

if [ $stage -le 0 ]; then
  $cmd $tc_args JOB=1:$nj $dir/log/extract_xvectors.JOB.log \
    python steps_tf/nnet_gen_embedding.py $gpu_opts \
    $sdata/JOB $nnetdir/$model_name ark,scp:$dir/xvector.JOB.ark,$dir/xvector.JOB.scp
fi

if [ $stage -le 1 ]; then
  # Be careful here: the speaker-level iVectors are now length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of iVectors for each speaker and length-normalizing"
  $cmd $dir/log/speaker_mean.log \
    ivector-normalize-length scp:$dir/xvector.scp  ark:- \| \
    ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
    ivector-normalize-length ark:- ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp
fi

exit 0;
}
