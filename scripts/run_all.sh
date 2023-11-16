#! /bin/bash
echo 'Run all forecast model start'

# run DCDense
echo 'DCDense Model train start'
./DCDense.sh
echo 'DCDense Model train end'

# run DLinear
echo 'DLinear Model train start'
./DLinear.sh
echo 'DLinear Model train end'

# run Formers
echo 'Formers Model train start'
./Formers.sh
echo 'Formers Model train end'

echo 'Run all forecast model end'