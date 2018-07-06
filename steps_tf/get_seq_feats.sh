#!/bin/bash
{
set -e

# Begin configuration section.
cmd=run.pl
# each archive has data-chunks off length randomly chosen between
# $min_frames_per_eg and $max_frames_per_eg.
compress=true
min_frames_per_chunk=50
max_frames_per_chunk=300
frames_per_archive=1000000
num_train_multiple=50  # multiple of total training frames for xvector training
num_valid_multiple=3   # multiple of total valid frames for validation
stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-eg <#frames;50>                 # The minimum number of frames per chunk that we dump"
  echo "  --max-frames-per-eg <#frames;200>                # The maximum number of frames per chunk that we dump"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
dir=$2

for f in $data/utt2num_frames $data/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1

mkdir -p $dir/info 
mkdir -p $dir/temp
temp=$dir/temp

echo $feat_dim > $dir/info/feat_dim
cp $data/utt2num_frames $dir/temp/utt2num_frames

if [ $stage -le 0 ]; then
  echo "$0: Preparing train and validation lists"
  # Pick a list of heldout utterances for validation egs
  num_utts=`wc -l $data/utt2spk | awk '{print $1}'`
  num_heldout_utts=`echo $num_utts | awk '{printf("%d", 0.1*$1)}'`
  awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $temp/valid_uttlist
  # The remaining utterances are used for training egs
  utils/filter_scp.pl --exclude $temp/valid_uttlist $temp/utt2num_frames > $temp/utt2num_frames.train
  utils/filter_scp.pl $temp/valid_uttlist $temp/utt2num_frames > $temp/utt2num_frames.valid
  # Create a mapping from utterance to speaker ID (an integer)
  awk -v id=0 '{print $1, id++}' $data/spk2utt > $temp/spk2int
  utils/sym2int.pl -f 2 $temp/spk2int $data/utt2spk > $temp/utt2int
  utils/filter_scp.pl $temp/utt2num_frames.train $temp/utt2int > $temp/utt2int.train
  utils/filter_scp.pl $temp/utt2num_frames.valid $temp/utt2int > $temp/utt2int.valid
fi

num_pdfs=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)
# The script assumes you've prepared the features ahead of time.
feats="scp,s,cs:utils/filter_scp.pl $temp/ranges.JOB $data/feats.scp |"
train_subset_feats="scp,s,cs:utils/filter_scp.pl $temp/train_subset_ranges.JOB $data/feats.scp |"
valid_feats="scp,s,cs:utils/filter_scp.pl $temp/valid_ranges.JOB $data/feats.scp |"

# first for the training data... work out how many archives.
num_train_frames=$(awk '{n += $2} END{print n}' <$temp/utt2num_frames.train)
num_valid_frames=$(awk '{n += $2} END{print n}' <$temp/utt2num_frames.valid)

echo $num_train_frames >$dir/info/num_frames.train
echo $num_valid_frames >$dir/info/num_frames.valid

if [ $stage -le 2 ]; then
  echo "$0: Allocating training examples"
  $cmd $dir/log/allocate_examples_train.log \
    steps_tf/allocate_egs.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --frames-per-archive=$frames_per_archive \
      --num-archives=$num_train_multiple \
      --utt2len-filename=$dir/temp/utt2num_frames.train \
      --utt2int-filename=$dir/temp/utt2int.train --egs-dir=$dir

  echo "$0: Allocating validation examples"
  $cmd $dir/log/allocate_examples_valid.log \
    steps_tf/allocate_egs.py \
      --prefix valid \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --randomize-chunk-length false \
      --frames-per-archive=$frames_per_archive \
      --num-archives=$num_valid_multiple \
      --utt2len-filename=$dir/temp/utt2num_frames.valid \
      --utt2int-filename=$dir/temp/utt2int.valid --egs-dir=$dir  || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: Generating training examples on disk"
  $cmd JOB=1:$num_train_multiple $dir/log/train_create_examples.JOB.log \
    nnet3-xvector-get-egs-feats --num-pdfs=$num_pdfs $temp/ranges.JOB \
    "$feats" ark,scp:$temp/feats.train.JOB.ark,$temp/feats.train.JOB.scp \
    ark:$temp/utt2label.JOB.train
  echo "$0: Generating validation examples on disk"
  $cmd JOB=1:$num_valid_multiple $dir/log/valid_create_examples.JOB.log \
    nnet3-xvector-get-egs-feats --num-pdfs=$num_pdfs $temp/valid_ranges.JOB \
    "$valid_feats" ark,scp:$dir/feats.valid.JOB.ark,$dir/feats.valid.JOB.scp \
    ark:$temp/utt2label.JOB.valid
fi

if [ $stage -le 4 ]; then
  echo "$0: Shuffling order of archives on disk"
  $cmd JOB=1:$num_train_multiple $dir/log/shuffle.JOB.log \
    copy-feats --compress=$compress \
    "scp:utils/shuffle_list.pl $temp/feats.train.JOB.scp |" \
    ark,scp:$dir/feats.train.JOB.ark,$dir/feats.train.JOB.scp \
    '&&' rm $temp/feats.train.JOB.scp $temp/feats.train.JOB.ark

fi

if [ $stage -le 5 ]; then
  for i in $(seq $num_train_multiple); do
    cat $dir/feats.train.$i.scp
  done > $dir/feats.train.scp
  
  for i in $(seq $num_train_multiple); do
    cat $temp/utt2label.$i.train
  done > $dir/utt2label.train
  
  for i in $(seq $num_valid_multiple); do
    cat $dir/feats.valid.$i.scp
  done > $dir/feats.valid.scp

  for i in $(seq $num_valid_multiple); do
    cat $temp/utt2label.$i.valid
  done > $dir/utt2label.valid

  echo $num_train_multiple > $dir/num_split.train
  echo $num_valid_multiple > $dir/num_split.valid

  wc $dir/feats.train.scp | awk '{print $1}' > $dir/num_samples.train
  wc $dir/feats.valid.scp | awk '{print $1}' > $dir/num_samples.valid
    
fi

echo "$0: Finished preparing training examples"
}
