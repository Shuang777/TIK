#!/bin/bash
{
set -e
set -o pipefail

# Begin configuration section.  
nj=4
cmd=run.pl
stage=0
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40

align_to_lats=false # optionally produce alignment in lattice format
 lats_decode_opts="--acoustic-scale=0.1 --beam=20 --latbeam=10"
 lats_graph_scales="--transition-scale=1.0 --self-loop-scale=0.1"

transform_dir=
cmvn_opts="--norm-vars=false"
splice_opts=
model_name=final.model.txt
# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
[ -f $srcdir/splice_opts ] && splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;

# Select default locations to model files
model=$dir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/L.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

if [ ! -z $transform_dir ]; then
  # we need to verify transforms for fmllr
  [ ! -f $transform_dir/trans.1 ] && echo "Cannot find $transform_dir/trans.1" && exit 1
  nj_orig=$(cat $transform_dir/num_jobs)
  if [ $nj -eq $nj_orig ]; then
    trans=trans.JOB
  else
    for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/$trans.ark,$dir/$trans.scp
    trans=trans.ark
  fi
fi

# Map oovs in reference transcription 
tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
graphs="ark:compile-train-graphs $dir/tree $dir/final.mdl $lang/L.fst \"$tra\" ark:- |"

echo "$0: aligning data '$data' using nnet/model '$srcdir', putting alignments in '$dir'"
# We could just use align-mapped in the next line, but it's less efficient as it compiles the
# training graphs one by one.
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    python steps_tf/nnet_forward.py --no-softmax --prior-counts $srcdir/ali_train_pdf.counts \
    --transform $transform_dir/$trans --verbose $sdata/JOB $srcdir/$model_name \| \
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl \
    "$graphs" ark:- "ark:|gzip -c >$dir/ali.JOB.gz"
fi

graphs="compile-train-graphs $lat_graph_scale $dir/tree $dir/final.mdl $lang/L.fst \"$tra\" ark:- |"

# Optionally align to lattice format (handy to get word alignment)
if [ "$align_to_lats" == "true" ]; then
  echo "$0: aligning also to lattices '$dir/lat.*.gz'"
  $cmd JOB=1:$nj $dir/log/align_lat.JOB.log \
    python steps_tf/nnet_forward.py --no-softmax --prior-counts $srcdir/ali_train_pdf.counts \
    --transform $transform_dir/$trans $sdata/JOB $srcdir/$model_name \| \
    latgen-faster-mapped $lat_decode_opts --word-symbol-table=$lang/words.txt $dir/final.mdl \
    "$graphs" ark:- "ark:|gzip -c >$dir/lat.JOB.gz"
fi

echo "$0: done aligning data."

}
