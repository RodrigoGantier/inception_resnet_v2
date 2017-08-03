'''
Created on 2 ago. 2017

@author: yasushishibe
'''


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import os
import sys
import argparse
import inception_resnet_v2
import dataset_utils
#import matplotlib.pyplot as plt

slim = tf.contrib.slim


def get_init_fn(model_path):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    #checkpoint_exclude_scopes=[]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        #print var
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..
      Args:
        input_width: Desired width of the image fed into the recognizer graph.
        input_height: Desired width of the image fed into the recognizer graph.
        input_depth: Desired channels of the image fed into the recognizer graph.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.
      Returns:
        Tensors for the node to feed JPEG data into, and the output of the
          preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image

def decoded_data(sess, tipe_list, len_list, batch_size, num_classes, tipe_dir, decoded_image_tensor, jpeg_data_tensor, ID_numbers):
    
    IDs    = tipe_list['ID'][len_list]
    labels = tipe_list['Label'][len_list]
    train_images = np.zeros([batch_size, 299, 299, 3])
    train_ground_truth = np.zeros([batch_size, num_classes])
    for num_, (ID_image, label_img) in enumerate(zip(IDs, labels)):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        image_path = os.path.join(tipe_dir, ID_image+'.jpg')
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # Then run it through the recognition network.
        train_images[num_, :, :, :] = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})
        train_ground_truth[num_, np.where(ID_numbers==label_img)[0][0] ] = 1
    return train_images, train_ground_truth

def main(_):
    
    model_path =   os.path.join(FLAGS.image_dir, 'inception_resnet_v2_2016_08_30.ckpt') 
    train_txt =    (FLAGS.image_dir+'/data_train_image.txt')
    valid_txt =    (FLAGS.image_dir+'/val.txt')
    train_dir =    (FLAGS.image_dir+'/train')
    valid_dir  =   (FLAGS.image_dir+'/test')
    summary_path = (FLAGS.image_dir+'/summary')
    
    if not tf.gfile.Exists(model_path):
        url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
        dataset_utils.download_and_uncompress_tarball(url, FLAGS.image_dir)
    if not tf.gfile.Exists(summary_path):
        tf.gfile.MakeDirs(summary_path)
    
    learning_rate = 0.01
    epoch=50
    input_height = 299
    input_width = 299
    input_depth = 3 
    input_mean = 128
    input_std = 128
    batch_size = 100
    num_classes = 100
    
    
    train_list = pd.read_table(train_txt, delimiter = ' ', names=['ID', 'Label', 'URL'])
    train_list = train_list.sample(frac=1).reset_index(drop=True)
    valid_list = pd.read_table(valid_txt, delimiter = ' ', names=['ID', 'Label', 'URL'])
    valid_list = valid_list.sample(frac=1).reset_index(drop=True)
    ID_numbers = pd.DataFrame(train_list.Label.unique())
    
    
    
    with tf.Graph().as_default():
        
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std)
        
        x = tf.placeholder(tf.float32, shape=(None, input_height, input_width, input_depth))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x, num_classes=num_classes, is_training=False)
        probabilities = tf.nn.softmax(logits)
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean)
        
        prediction = tf.argmax(probabilities, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        
        
        
        #init_fn = slim.assign_from_checkpoint_fn(
        #    os.path.join('/Users/yasushishibe/Desktop/baidu_dog', 'inception_resnet_v2_2016_08_30.ckpt'),
        #    slim.get_model_variables('InceptionResnetV2'))
        init = tf.global_variables_initializer()
        init_fn = get_init_fn(model_path)
       
        
        #variables_to_restore = slim.get_variables_to_restore(exclude=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"])
        # Create the saver which will be used to restore the variables.
        #restorer = tf.train.Saver(variables_to_restore)
           
        with tf.Session() as sess:
            
            merged = tf.summary.merge_all()
            train_writer =      tf.summary.FileWriter(summary_path + '/train', sess.graph)
            validation_writer = tf.summary.FileWriter(summary_path + '/validation')
            
            sess.run(init)
            #restorer.restore(sess, os.path.join('/Users/yasushishibe/Desktop/baidu_dog', 'inception_resnet_v2_2016_08_30.ckpt'))
            init_fn(sess)  
            
            for i in range(epoch):
                
                serie_1 = range(0, len(train_list), batch_size)
                serie_2 = range(batch_size, len(train_list), batch_size)
                
                for init, finish in zip(serie_1, serie_2):
                    resize_val, truth_val = decoded_data(sess,train_list, 
                                                         range(init,finish), batch_size, 
                                                         num_classes,train_dir, 
                                                         decoded_image_tensor, jpeg_data_tensor, 
                                                         ID_numbers)  
                    feed_dict={x: resize_val, y: truth_val}
                    train_summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, i)
                    
                    if init%(2*batch_size)==0:
                        feed_dict={x: resize_val, y: truth_val}
                        train_accuracy, cross_entropy_value = sess.run([accuracy, cross_entropy_mean], feed_dict=feed_dict)
                        print 'Training Accuracy: {}, Training Cross entropy: {}'.format(train_accuracy * 100, cross_entropy_value)
                        
                        resize_val, truth_val = decoded_data(sess,valid_list, 
                                                             range(init,finish), batch_size, 
                                                             num_classes,valid_dir, 
                                                             decoded_image_tensor, jpeg_data_tensor, 
                                                             ID_numbers) 
                        feed_dict={x: resize_val, y: truth_val}
                        valid_accuracy, valid_cross_entropy_value, validation_summary = sess.run([accuracy, cross_entropy_mean, merged], feed_dict=feed_dict)
                        validation_writer.add_summary(validation_summary, i)
                        print 'Training Accuracy: {}, Validation Cross entropy: {}'.format(valid_accuracy * 100, valid_cross_entropy_value)
                        
                        
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
            print sorted_inds[:5]
               
        print 'Finish'

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
