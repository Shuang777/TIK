#!/bin/bash
{
set -e

# Begin configuration section.
cmd=run.pl
cv_percent=10 # 
cv_spk=random   # in which way we split speakers; all / random / split
stage=0
nj=6          # This should be set to the maximum number of jobs you are
              # comfortable to run in parallel
num_utt_per_split=2000  # number of utterances per split
transdir=
cmvn_type=utt           # utt or sliding
cmvn_opts=
feat_type=raw           # raw, delta, lda, fmllr
srand=777
# End configuration section.
                            
echo "$0 $@"  # Print the command line for logging

. ./path.sh
. parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <feats-dir>"
  echo " e.g.: $0 data/train exp/swbd_feats"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
dir=$2

for f in $data/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  myutils/subset_data_dir_tr_cv.sh --cv-utt-percent $cv_percent \
    --spk $cv_spk $data $dir/train $dir/valid

  # shuffle the training scp
  cat $dir/train/feats.scp | utils/shuffle_list.pl --srand $srand > $dir/train/shuffle.feats.scp
fi

num_tr_utts=$(wc -l $dir/train/shuffle.feats.scp | awk '{print $1}')
num_tr_splits=$(python -c "from math import ceil; print int(ceil(1.0*$num_tr_utts / $num_utt_per_split))")
num_cv_utts=$(wc -l $dir/valid/feats.scp | awk '{print $1}')
num_cv_splits=$(python -c "from math import ceil; print int(ceil(1.0*$num_cv_utts / $num_utt_per_split))")

echo "Num of tr utts: $num_tr_utts"
echo "Num of tr splits: $num_tr_splits"
echo "Num of cv utts: $num_cv_utts"
echo "Num of cv splits: $num_cv_splits"

if [ $stage -le 1 ]; then
  echo "Preparing split scps"
  $cmd JOB=1:$num_tr_splits $dir/log/gen_tr_feats_scp.JOB.log \
    myutils/split_scp.pl -j $num_tr_splits JOB \
      $dir/train/shuffle.feats.scp $dir/train/split.JOB.scp

  $cmd JOB=1:$num_cv_splits $dir/log/gen_cv_feats_scp.JOB.log \
    myutils/split_scp.pl -j $num_cv_splits JOB \
      $dir/valid/feats.scp $dir/valid/split.JOB.scp
fi

echo $num_tr_splits > $dir/num_split.train
echo $num_cv_splits > $dir/num_split.valid

echo $cmvn_opts  > $dir/cmvn_opts # keep track of options to CMVN.
echo $cmvn_type > $dir/cmvn_type # keep track of type of CMVN

if [ $cmvn_type == sliding ]; then
  cmvn_feats_tr="apply-cmvn-sliding $cmvn_opts --center=true"
  cmvn_feats_cv="apply-cmvn-sliding $cmvn_opts --center=true"
elif [ $cmvn_type == utt ]; then
  cmvn_feats_tr="apply-cmvn $cmvn_opts --utt2spk=ark:$dir/train/utt2spk scp:$dir/train/cmvn.scp"
  cmvn_feats_cv="apply-cmvn $cmvn_opts --utt2spk=ark:$dir/valid/utt2spk scp:$dir/valid/cmvn.scp"
  [ ! -f $dir/train/cmvn.scp ] && echo "$dir/train/cmvn.scp not found" && exit 1
  [ ! -f $dir/valid/cmvn.scp ] && echo "$dir/valid/cmvn.scp not found" && exit 1
else
  echo "Wrong cmvn_type $cmvn_type" && exit 1
fi

echo "$0: feature type is $feat_type"
case $feat_type in
  raw) feats_tr="scp:$dir/train/split.JOB.scp"
         feats_cv="scp:$dir/valid/split.JOB.scp"
   ;;
  cmvn|traps) feats_tr="ark,s,cs:$cmvn_feats_tr scp:$dir/train/split.JOB.scp ark:- |"
       feats_cv="ark,s,cs:$cmvn_feats_cv scp:$dir/valid/split.JOB.scp ark:- |"
   ;;
  delta) feats_tr="ark,s,cs:$cmvn_feats_tr scp:$dir/train/split.JOB.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
         feats_cv="ark,s,cs:$cmvn_feats_cv scp:$dir/valid/split.JOB.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
   ;;
  lda|fmllr) feats_tr="ark,s,cs:$cmvn_feats_tr scp:$dir/train/split.JOB.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
       feats_cv="ark,s,cs:$cmvn_feats_cv scp:$dir/valid/split.JOB.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    cp $transdir/final.mat $dir
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -f $transdir/trans.1 ] && [ $feat_type == "fmllr" ]; then
  echo "$0: using transforms from $transdir"
  feats_tr="$feats_tr transform-feats --utt2spk=ark:$dir/train/utt2spk 'ark:cat $transdir/trans.*|' ark:- ark:- |"
  feats_cv="$feats_cv transform-feats --utt2spk=ark:$dir/valid/utt2spk 'ark:cat $transdir/trans.*|' ark:- ark:- |"
fi

if [ $stage -le 2 ]; then
  $cmd -tc $nj JOB=1:$num_tr_splits $dir/log/gen_tr_feats.JOB.log \
    copy-feats "$feats_tr" ark,scp:$dir/feats.train.JOB.ark,$dir/feats.train.JOB.scp

  $cmd -tc $nj JOB=1:$num_cv_splits $dir/log/gen_cv_feats.JOB.log \
    copy-feats "$feats_cv" ark,scp:$dir/feats.valid.JOB.ark,$dir/feats.valid.JOB.scp
fi

for i in `seq $num_tr_splits`; do
  cat $dir/feats.train.$i.scp
done > $dir/feats.train.scp

for i in `seq $num_cv_splits`; do
  cat $dir/feats.valid.$i.scp
done > $dir/feats.valid.scp

wc $dir/train/feats.scp | awk '{print $1}' > $dir/num_samples.train
wc $dir/valid/feats.scp | awk '{print $1}' > $dir/num_samples.valid

awk 'BEGIN {spk_id = 0} {print $1, spk_id; spk_id++;}' $dir/train/spk2utt > $dir/spk2id
for i in train valid; do
  awk 'NR==FNR {a[$1] = $2; next} {print $1, a[$2]}' $dir/spk2id $dir/$i/utt2spk \
    > $dir/utt2label.$i
done

feat_dim=$(feat-to-dim scp:$dir/feats.train.scp -)

echo $feat_dim > $dir/feat_dim

echo "$0: Finished preparing training examples"
}
