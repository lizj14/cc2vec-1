#!/bin/bash

# downloaded from https://raw.githubusercontent.com/rsennrich/wmt16-scripts/master/sample/translate.sh

#if [ "$#" -ne 5 ]; then
#    echo "usage: model testfile testoutput testref bleuoutfile"
#    exit 1
#fi

TEST='./data/jiang_ase_2017/test.3000.msg.predict.attention'
#TEST='./data/jiang_ase_2017/test.3000.diff.nmt'
TESTREF='./data/jiang_ase_2017/test.3000.msg'
BLEUOUT='./data/jiang_ase_2017/test.3000.predict.BLEU'


# the following is copied from validate.sh

## get BLEU
# path to moses decoder: https://github.com/moses-smt/mosesdecoder

mosesdecoder=../mosesdecoder
ref=$TESTREF

#./postprocess-dev.sh < $OUT > $OUT.postprocessed

$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $TEST >> $BLEUOUT
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $TEST | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "BLEU = $BLEU"
