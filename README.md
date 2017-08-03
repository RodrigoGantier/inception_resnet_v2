# inception_resnet_v2
This repository trains (fine tuning) an inception_resnet_v3 neural network with the last layer modified to recognize 100 species of dogs

To test you tensorflow configuration run the following command

python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"

You have to make sure to put the training data in the train folder, the validation data in the test folder

To run the program enter the following comand in your terminal:


python main.py --image_dir= 'Your training dir'
