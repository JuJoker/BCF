#! /bin/bash
echo 'Run all forecast model start'

# run DCDense
echo 'DCDense Model train start'
./DCDense.sh
echo 'DCDense Model train end'

# run DLinear
echo 'Seq2Seq Model train start'
./Seq2Seq.sh
echo 'Seq2Seq Model train end'

# run LSTM
echo 'LSTM Model train start'
./LSTM.sh
echo 'LSTM Model train end'

# run CNNLSTM
echo 'CNNLSTM Model train start'
./CNNLSTM.sh
echo 'CNNLSTM Model train end'

# run Formers
echo 'Formers Model train start'
./Formers.sh
echo 'Formers Model train end'

echo 'Run all forecast model end'