from subprocess import Popen, PIPE
import tempfile
import kaldiIO
import pickle
import shutil
import numpy
import os

DEVNULL = open(os.devnull, 'w')

## Data generator class for Kaldi
class dataGenerator:
    def __init__ (self, data, ali, 
                  transDir, exp, name, 
                  batchSize=256, 
                  seed=777, 
                  shuffle=False,
                  loop=False):
        self.data = data
        self.ali = ali
        self.exp = exp
        self.name = name
        self.batchSize = batchSize
        self.splice = 5
        self.loop = loop    # keep looping over dataset
        self.maxSplitDataSize = 1000 ## These many utterances are loaded into memory at once.

        self.tempDir = tempfile.mkdtemp(prefix='/data/exp/')
        aliPdf = self.tempDir + '/alipdf.' + self.name + '.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)
        self.labels = labels
       
        self.featDim = 440
        self.splitDataCounter = 0
        
        self.x = numpy.empty ((0, self.featDim))
        self.y = numpy.empty (0, dtype='uint32')
        self.batchPointer = 0

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)  # integer division

        cmd = 'cat ' + data + '/feats.scp | utils/shuffle_list.pl --srand ' + str(seed) + ' > ' + exp + '/shuffle.' + self.name + '.scp'
        Popen(cmd, shell=True).communicate()

        p1 = Popen (['apply-cmvn', '--utt2spk=ark:' + self.data + '/utt2spk',
                'scp:' + self.data + '/cmvn.scp',
                'scp:' + exp + '/shuffle.' + self.name + '.scp','ark:-'],
                stdout=PIPE)
        p2 = Popen (['splice-feats', 'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()
        p3 = Popen (['transform-feats', exp+'/final.mat', 'ark:-', 'ark:-'], stdin=p2.stdout, stdout=PIPE)
        p2.stdout.close()
        p4 = Popen (['transform-feats','--utt2spk=ark:' + self.data + '/utt2spk','ark:cat %s/trans.* |' % transDir,
            'ark:-', 'ark,scp:'+ self.tempDir + '/shuffle.' + self.name + '.ark,' + exp + '/' + self.name + '.scp'], stdin=p3.stdout).communicate()
        p3.stdout.close()

        split_scp_cmd = 'utils/split_scp.pl ' + exp + '/' + self.name + '.scp'
        for i in range(self.numSplit):
            split_scp_cmd += ' ' + self.tempDir + '/split.' + self.name + '.' + str(i) + '.scp'

        Popen (split_scp_cmd, shell=True).communicate()

        numpy.random.seed(seed)


    def getFeatDim(self):
        return self.featDim


    ## Clean-up label directory
    def __exit__ (self):
        shutil.rmtree(self.tempDir)
    

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1
            labels[line[0]] = [int(i) for i in line[1:]]
        return labels, numFeats
    

    ## Return a batch to work on
    def getNextSplitData (self):
#        p1 = Popen (['copy-feats', 'scp:' + self.tempDir.name + '/split.' + self.name + '.' + str(self.splitDataCounter) + '.scp', 'ark:-'], stdout=PIPE, stderr=DEVNULL)
        p1 = Popen (['splice-feats', '--print-args=false', '--left-context='+str(self.splice), '--right-context='+str(self.splice), 'scp:' + self.tempDir + '/split.' + self.name + '.' + str(self.splitDataCounter) + '.scp', 'ark:-'], stdout=PIPE, stderr=DEVNULL)

        featList = []
        labelList = []
        while True:
            uid, featMat = kaldiIO.readUtterance (p1.stdout)
            if uid == None:
                # no more utterance, return
                return (numpy.vstack(featList), numpy.hstack(labelList))
            if uid in self.labels:
                featList.append (featMat)
                labelList.append (self.labels[uid])


    def hasData(self):
    # has enough data for next batch
      if self.loop or self.splitDataCounter != self.numSplit:     # we always have data if in loop mode
          return True
      elif self.batchPointer + self.batchSize >= len(self.x):
          return False
      return True
        
            
    ## Retrive a mini batch
    def get_batch (self, feats_pl, labels_pl):
        # read split data until we have enough for this batch
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.loop and self.splitDataCounter == self.numSplit:
                # not loop mode and we arrive the end, do not read anymore
                return None

            x,y = self.getNextSplitData()
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.append (self.y[self.batchPointer:], y)
            self.batchPointer = 0

            ## Shuffle data
            randomInd = numpy.array(range(len(self.x)))
            numpy.random.shuffle(randomInd)
            self.x = self.x[randomInd]
            self.y = self.y[randomInd]

            self.splitDataCounter += 1
            if self.loop and self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
        
        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
        feed_dict = {
            feats_pl: xMini, 
            labels_pl: yMini
        }
        return feed_dict

    def get_batch_size(self):
        return self.batchSize
