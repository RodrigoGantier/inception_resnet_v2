import tensorflow as tf
import numpy as np
import sys
import os
import re
import pandas as pd
import argparse
import tarfile
import urllib.request
import inception_resnet_v2
from tensorflow.python.platform import gfile
import shutil
slim = tf.contrib.slim


def maybe_download_and_extract(dest_directory, data_url):
    """Download and extract model tar file.
      If the pretrained model we're using doesn't already exist, this function
      downloads it from the TensorFlow.org website and unpacks it into a directory.
      Args:
        data_url: Web location of the tar file containing the pretrained model.
        dest_directory: were to download the file
      """

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(os.path.split(filepath)[0]):
        print('Does not exist this path: "', os.path.dirname(filepath), '" will be crate', sep=' ')
        os.mkdir(os.path.dirname(filepath))

    if not os.path.exists(filepath):
        print('Does not exist the checkpoint file, it will be downloaded')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        stat_info = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, stat_info.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_image_lists(training_dir, testing_dir, validation_dir):
    """Builds a list of training, testing and validations images from the file system.
      Analyzes the sub folders in the training and validation image directory,
       and returns a data structure
      describing the lists of images for each label and their paths.
      Args:
        training_dir: String path to a folder containing sub folders with training images.
        testing_dir: String path to a folder containing testing of images.
        validation_dir: String path to a folder containing sub folders of validation images.
      Returns:
        Three pandas data frame containing, Image path and label
        training and validation. testing does not have labels.
      """

    training = pd.DataFrame(columns=['Files_name', 'labels'])
    testing = pd.DataFrame(columns=['Files_name'])
    validation = pd.DataFrame(columns=['Files_name', 'labels'])

    for i, directory in enumerate([training_dir, testing_dir, validation_dir]):

        sub_dirs = [x[0] for x in gfile.Walk(directory)]

        # The root directory comes first, so skip it.
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir:
                is_root_dir = False
                continue
            extension = 'jpg'
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == directory:
                continue
            tf.logging.info("Looking for images in '" + dir_name + "'")
            file_glob = os.path.join(directory, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                tf.logging.warning('No files found')
                continue
            files = np.array(file_list).reshape([-1, 1])
            labels = np.tile(re.sub(r'[^a-z0-9]+', ' ', dir_name.lower()), [len(file_list)]).reshape([-1, 1])
            files_labels = pd.DataFrame(np.append(files, labels, axis=1), columns=['Files_name', 'labels'])
            if i == 0:
                training = pd.concat([training, files_labels], axis=0, ignore_index=True)
            elif i == 1:
                testing = pd.concat([testing, pd.DataFrame(files, columns=['Files_name'])], axis=0)
            elif i == 2:
                validation = pd.concat([validation, files_labels], axis=0)

    id_numbers = pd.DataFrame(
        pd.concat([training['labels'], validation['labels']], axis=0, ignore_index=True).unique(),
        columns=['ID'])

    return training.sample(frac=1).reset_index(), testing.sample(frac=1).reset_index(), validation.sample(frac=1).reset_index(), id_numbers


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


def decoded_data(sess, type_list, range_list,
                 batch_size, num_classes,
                 decoded_image_tensor, jpeg_data_tensor,
                 id_numbers, input_height, input_width):

    ids = type_list.iloc[range_list]['Files_name']
    labels = type_list.iloc[range_list]['labels']
    train_images = np.zeros([batch_size, input_height, input_width, 3])
    train_ground_truth = np.zeros([batch_size, num_classes])
    for num_, (path_image, label_img) in enumerate(zip(ids, labels)):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        image_data = gfile.FastGFile(path_image, 'rb').read()

        # Then run it through the recognition network.
        train_images[num_, :, :, :] = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})
        train_ground_truth[num_, np.where(id_numbers == label_img)[0][0]] = 1

    return train_images, train_ground_truth


def decoded_data_for_test(sess, type_list, range_list,
                          batch_size, decoded_image_tensor, jpeg_data_tensor,
                          input_height, input_width):

    ids = type_list.iloc[range_list]['Files_name']
    train_images = np.zeros([batch_size, input_height, input_width, 3])
    for num_, path_image in enumerate(ids):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        image_data = gfile.FastGFile(path_image, 'rb').read()

        # Then run it through the recognition network.
        train_images[num_, :, :, :] = sess.run(decoded_image_tensor, {jpeg_data_tensor: image_data})

    return train_images


def get_init_fn(checkpoints_dir):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    # 'When restoring a checkpoint would ignore missing variables.'
    # return slim.assign_from_checkpoint_fn(checkpoints_dir, variables_to_restore, ignore_missing_vars=True)
    return slim.assign_from_checkpoint(checkpoints_dir, variables_to_restore)


def get_total_accuracy(sess, x, y, is_train, accuracy, type_list, batch_size, num_classes,
                       decoded_image_tensor, jpeg_data_tensor, id_numbers,
                       input_height, input_width):

    s_1 = range(0, len(type_list), batch_size)
    s_2 = range(batch_size, len(type_list), batch_size)
    total_accuracy = []
    for init, finish in zip(s_1, s_2):
        resize_val, truth_val = decoded_data(sess, type_list,
                                             range(init, finish), batch_size,
                                             num_classes,
                                             decoded_image_tensor, jpeg_data_tensor,
                                             id_numbers, input_height, input_width)
        feed_dict = {x: resize_val, y: truth_val, is_train: False}
        total_accuracy.append(sess.run(accuracy, feed_dict=feed_dict))
    return np.mean(total_accuracy)


def main(_):

    # parameters
    learning_rate = 0.001
    epoch = 50
    input_height = 299
    input_width = 299
    input_depth = 3
    input_mean = 128
    input_std = 128
    batch_size = 15
    num_classes = 100

    data_url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
    model_path = FLAGS.model_dir
    train_dir = FLAGS.train_dir
    test_dir = FLAGS.test_dir
    valid_dir = FLAGS.valid_dir
    summary_path = FLAGS.summary_dir
    os.mkdir(os.path.join(summary_path, "checkpoints"))
    checkpoints_path = os.path.join(summary_path, "checkpoints/inception_resnet_v2.ckpt")

    maybe_download_and_extract(FLAGS.image_dir, data_url)

    # Look at the folder structure, and create lists of all the images.
    train_list, test_list, valid_list, id_numbers = create_image_lists(train_dir, test_dir, valid_dir)

    with tf.Graph().as_default():
        # Sets the threshold for what messages will be logged, in this case it is set to 'INFO'
        tf.logging.set_verbosity(tf.logging.INFO)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(input_width,  input_height,
                                                                   input_depth,  input_mean,
                                                                   input_std)
        x = tf.placeholder(tf.float32, shape=(None, input_height, input_width, input_depth))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))
        is_train = tf.placeholder(tf.bool)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, _ = inception_resnet_v2.inception_resnet_v2(x, num_classes=num_classes, is_training=is_train)

        # Calculate the loss and create the optimizer:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        # State the metrics that you want to predict.
        prediction = tf.argmax(tf.nn.softmax(logits), 1)
        y_ = tf.argmax(y, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_), tf.float32))

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Loss', loss)
        tf.summary.scalar('Train accuracy', accuracy)

        # init_fn = get_init_fn(model_path)
        init_assign_op, init_feed_dict = get_init_fn(model_path)

        # saver function
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # init_fn(sess)
            sess.run(tf.global_variables_initializer())
            sess.run(init_assign_op, init_feed_dict)

            # If exists the checkpoint file, restore it
            if os.path.exists(os.path.join(os.path.dirname(checkpoints_path), "checkpoint")):
                saver.restore(sess, checkpoints_path)

            # marge all the summary data
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
            validation_writer = tf.summary.FileWriter(summary_path + '/validation')

            counter = 0
            for i in range(epoch):

                try:
                    # shuffle the order of the pictures
                    train_list = train_list.sample(frac=1).reset_index()
                    test_list = test_list.sample(frac=1).reset_index()
                except ValueError:
                    train_list = train_list
                    test_list = test_list

                s_1 = range(0, len(train_list), batch_size)
                s_2 = range(batch_size, len(train_list), batch_size)

                for init, finish in zip(s_1, s_2):
                    resize_val, truth_val = decoded_data(sess, train_list,
                                                         range(init, finish), batch_size,
                                                         num_classes,
                                                         decoded_image_tensor, jpeg_data_tensor,
                                                         id_numbers, input_height, input_width)
                    feed_dict = {x: resize_val, y: truth_val, is_train: True}
                    _, train_summary = sess.run([optimizer, merged], feed_dict=feed_dict)
                    counter += 1
                    train_writer.add_summary(train_summary, counter)
                    if init % (batch_size*10) == 0:
                        feed_dict = {x: resize_val, y: truth_val, is_train: False}
                        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                        rand_list = np.asarray(np.random.random(batch_size)*len(valid_list), dtype=np.int32)
                        resize_val, truth_val = decoded_data(sess, valid_list,
                                                             rand_list, batch_size,
                                                             num_classes,
                                                             decoded_image_tensor, jpeg_data_tensor,
                                                             id_numbers, input_height, input_width)
                        feed_dict = {x: resize_val, y: truth_val, is_train: False}
                        valid_accuracy, valid_summary = sess.run([accuracy, merged], feed_dict=feed_dict)
                        validation_writer.add_summary(valid_summary, counter)
                        print('Epoch {}, training accuracy: {:.2f} %, '
                              'Valid accuracy:{:.2f} %'.format(i+1, train_accuracy*100, valid_accuracy*100))
                        # save checkpoints
                        saver.save(sess, checkpoints_path)

            total_train_accuracy = get_total_accuracy(sess, x, y, is_train, accuracy, train_list, batch_size,
                                                      num_classes, decoded_image_tensor, jpeg_data_tensor, id_numbers,
                                                      input_height, input_width)

            total_valid_accuracy = get_total_accuracy(sess, x, y, is_train, accuracy, valid_list, batch_size,
                                                      num_classes, decoded_image_tensor, jpeg_data_tensor, id_numbers,
                                                      input_height, input_width)

            print('Total Training accuracy: {:.2f} %, Total Validation accuracy: {:.2f} %'.format(
                total_train_accuracy*100, total_valid_accuracy*100))

            prediction_path_and_labels = pd.DataFrame(columns=['ID', 'Files_name'])
            s_1 = range(0, len(test_list), batch_size)
            s_2 = range(batch_size, len(test_list), batch_size)
            for init, finish in zip(s_1, s_2):
                resize_val = decoded_data_for_test(sess, test_list,
                                                   range(init, finish), batch_size,
                                                   decoded_image_tensor, jpeg_data_tensor,
                                                   input_height, input_width)
                feed_dict = {x: resize_val, is_train: False}
                new_data = pd.DataFrame(id_numbers.ID.iloc[sess.run(prediction, feed_dict=feed_dict)])
                new_data = new_data.assign(Files_name=test_list.Files_name.iloc[range(init, finish)].values)
                prediction_path_and_labels = pd.concat([prediction_path_and_labels, new_data], ignore_index=True)

            main_path = os.path.split(test_list.Files_name.iloc[0])[0]
            main_path = os.path.split(main_path)[0]
            for label, image_dir in prediction_path_and_labels.values.tolist():

                image_path = os.path.join(main_path, label)
                if not os.path.exists(image_path):
                    os.mkdir(image_path)

                shutil.copy(image_dir, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/media/rodrigo/3FA1-EDBD/baidu_dogs',
        help='Path to folders of Model checkpoint.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/media/rodrigo/3FA1-EDBD/baidu_dogs/data/SSD-Square-croptrain',
        help='Path to folders of train_dir.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/media/rodrigo/3FA1-EDBD/baidu_dogs/data/SSD-Square-croptest',
        help='Path to folders of test_dir.'
    )
    parser.add_argument(
        '--valid_dir',
        type=str,
        default='/media/rodrigo/3FA1-EDBD/baidu_dogs/data/SSD-Square-cropval',
        help='Path to folders of valid_dir.'
    )
    parser.add_argument(
        '--summary_dir',
        type=str,
        default='/media/rodrigo/3FA1-EDBD/baidu_dogs/summary',
        help='Path to folders of summary_dir.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=40,
        help='How many training steps to run before ending.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
