# inception_resnet_v2
This repository trains (fine tuning) an inception_resnet_v3 neural network with the last layer modified to recognize 100 species of dogs

To test you tensorflow configuration run the following command

python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
