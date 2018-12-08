# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.

# pylint: disable=line-too-long
r"""Simple transfer learning with image modules from TensorFlow Hub.

This example shows how to train an image classifier based on any
TensorFlow Hub module that computes image feature vectors. By default,
it uses the feature vectors computed by Inception V3 trained on ImageNet.
See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
for more options.

The top layer receives as input a 2048-dimensional vector (assuming
Inception V3) for each image. We train a softmax layer on top of this
representation. If the softmax layer contains N labels, this corresponds
to learning N + 2048*N model parameters for the biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. (For a working example,
download http://download.tensorflow.org/example_images/flower_photos.tgz
and run  tar xzf flower_photos.tgz  to unpack it.)

Once your images are prepared, and you have pip-installed tensorflow-hub and
a sufficiently recent version of tensorflow, you can run the training with a
command like this:

```bash
python retrain.py --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the tensorflow/examples/label_image sample code.

By default this script will use the highly accurate, but comparatively large and
slow Inception V3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--tfhub_module` flag with a
Mobilenet model. For more information on Mobilenet, see
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

For example:

Run floating-point version of Mobilenet:

```bash
python retrain.py --image_dir ~/flower_photos \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1
```

Run Mobilenet, instrumented for quantization:

```bash
python retrain.py --image_dir ~/flower_photos/ \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1
```

These instrumented models can be converted to fully quantized mobile models via
TensorFlow Lite.

There are different Mobilenet models to choose from, with a variety of file
size and latency options.
  - The first number can be '100', '075', '050', or '025' to control the number
    of neurons (activations of hidden layers); the number of weights (and hence
    to some extent the file size and speed) shrinks with the square of that
    fraction.
  - The second number is the input image size. You can choose '224', '192',
    '160', or '128', with smaller sizes giving faster speeds.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

To use with Tensorflow Serving, run this tool with --saved_model_dir set
to some increasingly numbered export location under the model base path, e.g.:

```bash
python retrain.py (... other args as before ...) \
    --saved_model_dir=/tmp/saved_models/$(date +%s)/
tensorflow_model_server --port=9000 --model_name=my_image_classifier \
    --model_base_path=/tmp/saved_models/
```
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


import h5py
from PIL import Image, ImageOps
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import cv2
import skimage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

#Begin scaffolding. Comment out all unnecessary code

def noisy(noise_typ,image): 

    if noise_typ == "gauss": 

      row,col,ch= 1024, 1024, 1 

      mean = 0 

      var = 0.1 

      sigma = var**0.5 

      gauss = np.random.normal(mean,sigma,(row,col,ch)) 

      gauss = gauss.reshape(row,col,ch) 

      noisy = image + gauss 

      return noisy 

    elif noise_typ == "s&p": 

      row,col,ch = 1024, 1024, 1 

      s_vs_p = 0.5 

      amount = 0.004 

      out = np.copy(image) 

      # Salt mode 

      num_salt = np.ceil(amount * image.size * s_vs_p) 

      coords = [np.random.randint(0, i - 1, int(num_salt)) 

              for i in image.shape] 

      out[coords] = 1 
      # Pepper mode 
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p)) 
      coords = [np.random.randint(0, i - 1, int(num_pepper)) 
              for i in image.shape] 
      out[coords] = 0 
      return out 
    elif noise_typ == "poisson": 

      vals = len(np.unique(image)) 

      vals = 2 ** np.ceil(np.log2(vals)) 

      noisy = np.random.poisson(image * vals) / float(vals) 

      return noisy 

    elif noise_typ =="speckle": 

      row,col,ch = 1024, 1024, 1 

      gauss = np.random.randn(row,col,ch) 

      gauss = gauss.reshape(row,col,ch) 

      noisy = image + image * gauss 

      return noisy 

 


#sets the kth bit to 0. (CCD images have 16 bits. We'll turn off 15 and 16)
def turnOffK(n,k):
    
    # k must be greater than 0
    if (k <= 0):
        return n
    
    # Do & of n with a number
    # with all set bits except
    # the k'th bit
    return (n & ~(1 << (k - 1)))


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def shear2(a, strength=1, shift_axis=0, increase_axis=1, edges='clip'): 

    indices = np.indices(a.shape) 

    indices[shift_axis] -= strength * indices[increase_axis] 

    indices[shift_axis] %= a.shape[shift_axis] 

    res = a[tuple(indices)] 

    if edges == 'clip': 

        res[indices[shift_axis] < 0] = 0 

        res[indices[shift_axis] >= a.shape[shift_axis]] = 0 

    return res

def add_gaussian_noise(image_in, noise_sigma): 

    temp_image = np.float64(np.copy(image_in)) 
    h = temp_image.shape[0] 
    w = temp_image.shape[1] 
    noise = np.random.randn(h, w) * noise_sigma 
    noisy_image = np.zeros(temp_image.shape, np.float64) 
    if len(temp_image.shape) == 2: 
        noise[0:1023,0:400] = 0 
        noise[0:1023,700:1023] = 0 
        noise[700:1023,0:1023] = 0 
        noise[0:300,0:1023] = 0 
        noisy_image = temp_image + noise 
    else: 
        noisy_image[:,:,0] = temp_image[:,:,0] + noise 
        noisy_image[:,:,1] = temp_image[:,:,1] + noise 
        noisy_image[:,:,2] = temp_image[:,:,2] + noise 

 
 

    return noisy_image
def backgroundReader(filename):
 file = h5py.File(filename, 'r')
 dataset = file['/NotHits/']
 background = dataset[0]
 background = turnOffK(background, 15)
 background = turnOffK(background, 16)
 return background

def imageAugmentor(image, eID, Parula):
    eIDStack = []
    imageStack = []
      
    bone = cm.get_cmap('bone')
    reuse = bone(image)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'bone')
    reuse = np.rot90(image)
    reuse = np.rot90(reuse)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'rotated')

    reuse = []
    reuse = shear2(image)
    #sheared_image[:, i] = np.roll(image[:, i], i)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    eIDStack.append(str(eID) + 'sheared')
    imageStack.append(reuse)


    hot = cm.get_cmap('hot')
    reuse = np.rot90(image)
    reuse = np.rot90(reuse)
    reuse = hot(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'hot')

    reuse = noisy('s&p', image)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'salty')


    reuse = noisy('poisson', image)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'poisson')

    reuse = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'gaussian')

    reuse = np.fliplr(image)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'flip1')
    
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.01,
                                 dtype=np.float64)
    reuse = tf.add(image, noise)
    reuse = tf.Session().run(reuse)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'morenoise')
    
    reuse = np.sqrt(image)
    reuse = Parula(reuse)
    reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'sqrt')

    reuse = add_gaussian_noise(image, 0.025) 
    reuse = Parula(reuse) 
    reuse = np.uint8(reuse*255) 
    imageStack.append(reuse) 
    eIDStack.append(str(eID) + 'gaussianTest') 
    
    
    """"""""""""
    image = Parula(image)
    image = np.uint8(image*255)
    """"""""""""
    reuse = tf.image.flip_up_down(image)
    reuse = tf.Session().run(reuse)
#            flipImage2 = Parula(flipImage2)
#            flipImage2 = np.uint8(flipImage2*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'flip2')

    reuse = tf.image.flip_left_right(image)
    reuse = tf.Session().run(reuse)
#            flipImage3 = Parula(flipImage3)
#            flipImage3 = np.uint8(flipImage3*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'flip3')

    reuse  = tf.image.random_flip_up_down(image)
    reuse = tf.Session().run(reuse)
#            flipImage4 = Parula(flipImage4)
#            flipImage4 = np.uint8(flipImage4*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'flip4')


    reuse  = tf.image.random_flip_left_right(image)
    reuse = tf.Session().run(reuse)
#            flipImage5 = Parula(flipImage5)
#            flipImage5 = np.uint8(flipImage5*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'flip5')

    reuse = tf.image.rot90(image, k=1)
    reuse = tf.Session().run(reuse)
#            reuse = Parula(reuse)
#            reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'rot90')


    reuse = tf.image.rot90(image, k=2)
    reuse = tf.Session().run(reuse)
#            reuse = Parula(reuse)
#            reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'rot180')
    #original_size = [1024, 1024, 1]
#            crop_size = [512, 512, 1]
#            seed = np.random.randint(1234)
#            reuse = tf.random_crop(reuse, size = crop_size, seed = seed)
#            reuse = tf.images.resize_images(reuse, size = original_size)

    reuse = tf.image.pad_to_bounding_box(image, 100, 0, 1024 + 100, 1024 + 100)
    reuse = tf.image.crop_to_bounding_box(reuse, 0, 0, 1024, 1024)
    reuse = tf.Session().run(reuse)
    reuse[0 == reuse] = 100
#            reuse = Parula(reuse)
#            reuse = np.uint8(reuse*255)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'translate1')
    
    
    reuse = tf.image.pad_to_bounding_box(image, 0, 0, 1024 + 100, 1024 + 100)
    reuse = tf.image.crop_to_bounding_box(reuse, 100, 100, 1024, 1024)
    reuse = tf.Session().run(reuse)
    reuse[0 == reuse] = 100
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'translate2')
    
    reuse = tf.image.pad_to_bounding_box(image, 0, 0, 1024 + 200, 1024 + 200)
    reuse = tf.image.crop_to_bounding_box(reuse, 200, 200, 1024, 1024)
    reuse = tf.Session().run(reuse)
    reuse[0 == reuse] = 100
    reuse = np.fliplr(reuse)
    imageStack.append(reuse)
    eIDStack.append(str(eID) + 'translate3')

    return imageStack, eIDStack

#H5reader takes in a file path to an H5 file
def H5reader(filename):
    global background
    color = pd.read_csv("/home/ctonline/MachineLearning/Parula.txt", header = None)
    color = np.array(color)
    Parula = LinearSegmentedColormap.from_list('parula', color)
    imageStack = []
    eIDStack = []
    file    = h5py.File(filename, 'r')
    if filename == "/home/ctonline/MachineLearning/FEL/Hits.hdf5":
      print("File Hits Opened")
      dataset  = file['/FoundHits/']
      eID = file['/FoundHitsBatchID/']
    else:
      dataset  = file['/NotHits/']
      eID = file['/NotHitsBatchID/']
    #Loop over all images in the H5 file (100)
    for i in range(len(dataset)):
#        background = dataset[0]
        image = dataset[i]

        #Turning off bits
        image = turnOffK(image, 15)
        image = turnOffK(image, 16)
#
#        background = turnOffK(background, 15)
#        background = turnOffK(background, 16)


        #Background subtraction
        image = image - background

        # We have to prevent underflow for when we convert to unisgned ints
        image[0 > image] = 0


        image = np.sqrt(image)

        #Normalizing the image for color mapping
        #image = rescale_linear(image, 0.0035713999999999997, 0.9988299999999999)


        image = rescale_linear(image, 0.0035713999999999997, 0.89)
        # Gotta modify the images images just slightly, so the CNN has a larger training set


        if eID[i] == 21805667:
            pass
        # This specific image has some problems
        if filename == "/home/ctonline/MachineLearning/FEL/Hits.hdf5":
            functionImageStack, functioneIDStack = imageAugmentor(image, eID[i], Parula)
            imageStack.extend(functionImageStack)
            eIDStack.extend(functioneIDStack)
        elif i % 50 == 0:
            print(i)
            reuse = image*0
            reuse = Parula(reuse)
            reuse = np.uint8(reuse*255)
            imageStack.append(reuse)
            eIDStack.append(str(eID) + 'blank')
            functionImageStack, functioneIDStack = imageAugmentor(image, eID[i], Parula)
            imageStack.extend(functionImageStack)
            eIDStack.extend(functioneIDStack)
        else:
            pass
        
    
        #Preparing for jpg conversion
   #     image = Parula(image)
    #    image = np.uint8(image*255)
        
        # Gotta modify the Hits images just slightly
##        image = Image.fromarray(image)
##        image = image.convert('RGB')
##        image = image.rotate(270)

#        #Adding the converted raw images to a list
        image = Parula(image)
        image = np.uint8(image*255)
        imageStack.append(image)
        eIDStack.append(eID[i])

        if not eID[i] == 21805667:
           imageStack.append(image)
           eIDStack.append(eID[i])
        else:
           pass
 
    file.close()

    return imageStack, eIDStack



def main(_):
 #with tf.device('/device:GPU:0'):
    global listOfHits
    global listOfMisses
    retraining_lists = create_retraining_lists(listOfHits, listOfMisses, FLAGS.testing_percentage, FLAGS.validation_percentage)
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)


    # Prepare necessary directories that can be used during training
    prepare_file_system()

    # Look at the folder structure, and create lists of all the images.

    class_count = len(retraining_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + hitFile)

    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         hitFile +
                         ' - multiple classes are needed for classification.')

    do_distort_images = should_distort_images(
          FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
          FLAGS.random_brightness)

    # Set up the pre-trained graph.
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
    create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
        ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=True)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Initialize all weights: for the module to their pretrained values,
    # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)


        if do_distort_images:
          # We will be applying distortions, so set up the operations we'll need.
          (distorted_jpeg_data_tensor,
           distorted_image_tensor) = add_input_distortions(
               FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
               FLAGS.random_brightness, module_spec)
        else:
          # We'll make sure we've calculated the 'bottleneck' image summaries and
          # cached them on disk.
          cache_bottlenecks_for_retraining(sess, retraining_lists,
                            FLAGS.bottleneck_dir, jpeg_data_tensor,
                            decoded_image_tensor, resized_image_tensor,
                            bottleneck_tensor, FLAGS.tfhub_module)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()




        """ We'll get to this part later """

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
          # Get a batch of input bottleneck values, either calculated fresh every
          # time with distortions applied, or from the cache stored on disk.
          if do_distort_images:
            (train_bottlenecks,
             train_ground_truth) = get_random_distorted_bottlenecks_for_retraining(
                 sess, retraining_lists, FLAGS.train_batch_size, 'training', distorted_jpeg_data_tensor,
                 distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
          else:
            (train_bottlenecks,
             train_ground_truth, _) = get_random_cached_bottlenecks_for_retraining(
                 sess, retraining_lists, FLAGS.train_batch_size, 'training',
                 FLAGS.bottleneck_dir, jpeg_data_tensor,
                 decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                 FLAGS.tfhub_module)
          # Feed the bottlenecks and ground truth into the graph, and run a training
          # step. Capture training summaries for TensorBoard with the `merged` op.
          train_summary, _ = sess.run(
              [merged, train_step],
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
          train_writer.add_summary(train_summary, i)

          # Every so often, print out how well the graph is training.
          is_last_step = (i + 1 == FLAGS.how_many_training_steps)
          if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                            (datetime.now(), i, train_accuracy * 100))
            tf.logging.info('%s: Step %d: Cross entropy = %f' %
                            (datetime.now(), i, cross_entropy_value))
            # TODO: Make this use an eval graph, to avoid quantization
            # moving averages being updated by the validation set, though in
            # practice this makes a negligable difference.
            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks_for_retraining(
                    sess, retraining_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.tfhub_module))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (datetime.now(), i, validation_accuracy * 100,
                             len(validation_bottlenecks)))

          # Store intermediate results
          intermediate_frequency = FLAGS.intermediate_store_frequency

          if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
              and i > 0):
            # If we want to do an intermediate save, save a checkpoint of the train
            # graph, to restore into the eval graph.
            train_saver.save(sess, CHECKPOINT_NAME)
            intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                      'intermediate_' + str(i) + '.pb')
            tf.logging.info('Save intermediate result to : ' +
                            intermediate_file_name)
            save_graph_to_file(intermediate_file_name, module_spec,
                               class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, CHECKPOINT_NAME)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        run_final_eval(sess, module_spec, class_count, retraining_lists,
                       jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                       bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as
        # constants.
        tf.logging.info('Save final result to : ' + FLAGS.output_graph)
        if wants_quantization:
          tf.logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(FLAGS.output_graph, module_spec, class_count)
        with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
          f.write('\n'.join(retraining_lists.keys()) + '\n')

        if FLAGS.saved_model_dir:
          export_model(module_spec, class_count, FLAGS.saved_model_dir)


def create_module_graph(module_spec):
  """Creates a graph and loads Hub Module into it.

  Args:
    module_spec: the hub.ModuleSpec for the image module being used.

  Returns:
    graph: the tf.Graph that was created.
    bottleneck_tensor: the bottleneck values output by the module.
    resized_input_tensor: the input images, resized as expected by the module.
    wants_quantization: a boolean, whether the module has been instrumented
      with fake quantization ops.
  """
  height, width = hub.get_expected_image_size(module_spec)
  with tf.Graph().as_default() as graph:
    resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
    m = hub.Module(module_spec)
    bottleneck_tensor = m(resized_input_tensor)
    wants_quantization = any(node.op in FAKE_QUANT_OPS
                             for node in graph.as_graph_def().node)
  return graph, bottleneck_tensor, resized_input_tensor, wants_quantization

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)




def run_final_eval(train_session, module_spec, class_count, retraining_lists,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):
  """Runs a final evaluation on an eval graph using the test data set.

  Args:
    train_session: Session for the train graph with the tensors below.
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes
    image_lists: OrderedDict of training images for each label.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_image_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
  """
  test_bottlenecks, test_ground_truth, test_filenames = (
      get_random_cached_bottlenecks_for_retraining(train_session, retraining_lists,
                                    FLAGS.test_batch_size,
                                    'testing', FLAGS.bottleneck_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor,
                                    bottleneck_tensor, FLAGS.tfhub_module))

  (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
   prediction) = build_eval_session(module_spec, class_count)
  test_accuracy, predictions = eval_session.run(
      [evaluation_step, prediction],
      feed_dict={
          bottleneck_input: test_bottlenecks,
          ground_truth_input: test_ground_truth
      })
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                  (test_accuracy * 100, len(test_bottlenecks)))

  if FLAGS.print_misclassified_test_images:
    tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
    for i, test_filename in enumerate(test_filenames):
      if predictions[i] != test_ground_truth[i]:
        tf.logging.info('%70s  %s' % (test_filename,
                                      list(retraining_lists.keys())[predictions[i]]))


def build_eval_session(module_spec, class_count):
  """Builds an restored eval session without train operations for exporting.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes

  Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
  """
  # If quantized, we need to create the correct eval graph for exporting.
  eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (
      create_module_graph(module_spec))

  eval_sess = tf.Session(graph=eval_graph)
  with eval_graph.as_default():
    # Add the new layer for exporting.
    (_, _, bottleneck_input,
     ground_truth_input, final_tensor) = add_final_retrain_ops(
         class_count, FLAGS.final_tensor_name, bottleneck_tensor,
         wants_quantization, is_training=False)

    # Now we need to restore the values from the training graph to the eval
    # graph.
    tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

    evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                      ground_truth_input)

  return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
          evaluation_step, prediction)


def save_graph_to_file(graph_file_name, module_spec, class_count):
  """Saves an graph to file, creating a valid quantized one if necessary."""
  sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
  graph = sess.graph

  output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

  with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())

def export_model(module_spec, class_count, saved_model_dir):
  """Exports model for serving.

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: The number of classes.
    saved_model_dir: Directory in which to save exported model and variables.
  """
  # The SavedModel should hold the eval graph.
  sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
  with sess.graph.as_default() as graph:
    tf.saved_model.simple_save(
        sess,
        saved_model_dir,
        inputs={'image': in_image},
        outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
    )


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          quantize_layer, is_training):
  """Adds a new softmax and fully-connected layer for training and eval.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
        recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    quantize_layer: Boolean, specifying whether the newly added layer should be
        instrumented for quantization with TF-Lite.
    is_training: Boolean, specifying whether the newly add layer is for training
        or eval.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
  assert batch_size is None, 'We want to work with arbitrary batch size.'
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[batch_size, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(
        tf.int64, [batch_size], name='GroundTruthInput')

  # Organizing the following ops so they are easier to see in TensorBoard.
  layer_name = 'final_retrain_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)
      layer_weights = tf.Variable(initial_value, name='final_weights')
      variable_summaries(layer_weights)

    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)

    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

  # The tf.contrib.quantize functions rewrite the graph in place for
  # quantization. The imported model graph has already been rewritten, so upon
  # calling these rewrites, only the newly added final layer will be
  # transformed.
  if quantize_layer:
    if is_training:
      tf.contrib.quantize.create_training_graph()
    else:
      tf.contrib.quantize.create_eval_graph()

  tf.summary.histogram('activations', final_tensor)

  # If this is an eval graph, we don't need to add loss ops or an optimizer.
  if not is_training:
    return None, None, bottleneck_input, ground_truth_input, final_tensor

  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, ground_truth_tensor)
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, module_spec):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
    module_spec: The hub.ModuleSpec for the image module being used.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """
  input_height, input_width = hub.get_expected_image_size(module_spec)
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(shape=[],
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(shape=[],
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))

def add_jpeg_decoding(module_spec):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    module_spec: The hub.ModuleSpec for the image module being used.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  input_height, input_width = hub.get_expected_image_size(module_spec)
  #input_height, input_width = 1024, 1024
  input_depth = hub.get_num_image_channels(module_spec)
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image


def prepare_file_system():
  # Set up the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def getImage(batchID):
    global listOfHits
    global listOfMisses
    global imageStackOfHits
    global imageStackOfMisses
    
    if batchID in listOfMisses:
        return(imageStackOfMisses[listOfMisses.index(batchID)])
    elif batchID in listOfHits:
        return(imageStackOfHits[listOfHits.index(batchID)])
    else:
        return "Image not found"


def create_retraining_lists(listOfHits, listOfMisses, testing_percentage, validation_percentage):

    
    # TODO: Pass in two eIDStacks
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

    Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
    """
    #  if not tf.gfile.Exists(image_dir):
    #    tf.logging.error("Image directory '" + image_dir + "' not found.")
    #    return None

    result = collections.OrderedDict()
    #  sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    #  # The root directory comes first, so skip it.
    #  is_root_dir = True
    #  for sub_dir in sub_dirs:
    #    if is_root_dir:
    #      is_root_dir = False
    #      continue
    #    extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
    #                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg']))
    #    file_list = []

    #In our case, the dir_name will be "Hits"
    dir_name = 'Hits'
          #dir_name = os.path.basename(sub_dir) #Get the name of current directory
    #    if dir_name == image_dir:
    #      continue
    #    tf.logging.info("Looking for images in '" + dir_name + "'")
    #    for extension in extensions:
    #      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
    #file_list.extend(tf.gfile.Glob(file_glob)) #Adding file paths
    #    if not file_list:
    #      tf.logging.warning('No files found')
    #      continue
    #    if len(file_list) < 20:
    #      tf.logging.warning(
    #          'WARNING: Folder has less than 20 images, which may cause issues.')
    #    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
    #      tf.logging.warning(
    #          'WARNING: Folder {} has more than {} images. Some images will '
    #          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = 'Hits'
        #label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []


    #Going have to be extra careful to not ruin everything

    #for file_name in file_list:
    for eID in range(np.size(listOfHits)):
      base_name = listOfHits[eID]
        #base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', str(listOfHits[eID]))
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
    dir_name = 'Misses'
   
    label_name = 'Misses'
    training_images = []
    testing_images = []
    validation_images = []


    #for file_name in file_list:
    for eID in range(np.size(listOfMisses)):
      base_name = listOfMisses[eID]

      hash_name = re.sub(r'_nohash_.*$', '', str(listOfMisses[eID]))
  
      hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
#Now we have a ordered dictionary of hits and misses
    return result


def get_eID(retraining_lists, label_name, index, category):
  # In the entire training_list, just give me the ones that correspond to the label (Hits or Misses)
  label_lists = retraining_lists[label_name]
  # Then, out of that list, just give me the ones that correspond to the category we're working with (validation, training, test)
  category_list = label_lists[category]
  # Finally, just give me one of the images that correspond to this index
  mod_index = index % len(category_list)
  eID = category_list[mod_index]

  return eID

"""The following functions all call each other!"""



def get_random_cached_bottlenecks_for_retraining(sess, retraining_lists, how_many, category,
                                  bottleneck_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    module_name: The name of the image module being used.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  class_count = len(retraining_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(retraining_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)

      eID = get_eID(retraining_lists, label_name, image_index, category)
      bottleneck = get_or_create_bottleneck_for_retraining(
          sess, retraining_lists, label_name, image_index, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, module_name)
      bottlenecks.append(bottleneck)
      ground_truths.append(label_index)
      filenames.append(eID)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(retraining_lists.keys()):
      for image_index, eID in enumerate(
          retraining_lists[label_name][category]):
        eID = get_eID(retraining_lists, label_name, image_index, category)
        bottleneck = get_or_create_bottleneck_for_retraining(
            sess, retraining_lists, label_name, image_index, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name)
        bottlenecks.append(bottleneck)
        ground_truths.append(label_index)
        filenames.append(eID)
  return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks_for_retraining(
    sess, retraining_lists, how_many, category, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(retraining_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(retraining_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    eID = get_eID(retraining_lists, label_name, image_index, category)
   # if not tf.gfile.Exists(str(eID)):
    #  tf.logging.fatal('File does not exist %s', eID)
    image_data = getImage(eID)
    jpeg_data = cv2.imencode('.jpg', image_data)[1].tostring()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    bottlenecks.append(bottleneck_values)
    ground_truths.append(label_index)
  return bottlenecks, ground_truths





def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Then run it through the recognition network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def create_bottleneck_file_for_retraining(bottleneck_path, retraining_lists, label_name, index,
                           category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  #Okay, let's try to make sense of what's going on here,
  #First, the bottleneck path is passed in and printed out
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)

  #Then, we get an image path, this literally just the image directory
  eID = get_eID(retraining_lists, label_name, index, category)
  #Of course, we don't have paths to pass in
                              
  image_data = getImage(eID)
  image_data = cv2.imencode('.jpg', image_data)[1].tostring()
#  image_data = tf.cast(image, tf.float32)
  #Here, we just pass in our array
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (eID,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_path_to_bottleneck_dir(retraining_lists, label_name, index, bottleneck_dir, category):
  """Returns a path to an image for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in retraining_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = retraining_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)


  mod_index = index % len(category_list)
  eID = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(str(bottleneck_dir), str(sub_dir), str(eID))
  if not os.path.exists(bottleneck_dir):
      print("The bottleneck path doesn't even exiet")
      return -1
  else:
      pass
  return full_path


def get_bottleneck_path_for_retraining(retraining_lists, label_name, index, bottleneck_dir,
                        category, module_name):
  """Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    module_name: The name of the image module being used.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  module_name = (module_name.replace('://', '~')  # URL scheme.
                 .replace('/', '~')  # URL and Unix paths.
                 .replace(':', '~').replace('\\', '~'))  # Windows paths.
  #Label name is just the type of image (hit or miss)
  #category is either the test, validation, or training set
  return get_path_to_bottleneck_dir(retraining_lists, label_name, index, bottleneck_dir,
                        category) + '_' + module_name + '.txt'


def cache_bottlenecks_for_retraining(sess, retraining_lists, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    module_name: The name of the image module being used.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  #image_lists is the list of all images, obviously

  for label_name, label_lists in retraining_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck_for_retraining(
            sess, retraining_lists, label_name, index, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, module_name)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')


def get_or_create_bottleneck_for_retraining(sess, retraining_lists, label_name, index,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: OrderedDict of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of which set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    module_name: The name of the image module being used.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
  label_lists = retraining_lists[label_name]
  sub_dir = label_lists['dir'] # Just creating a subdirectory in /tmp/bottlneck/$label for the bottlenecks!
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path_for_retraining(retraining_lists, label_name, index,
                                        bottleneck_dir, category, module_name)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file_for_retraining(bottleneck_path, retraining_lists, label_name, index,
                           category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file_for_retraining(bottleneck_path, retraining_lists, label_name, index,
                           category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values



# We're treating the HDf5 files as global, so we don't have to pass 4 lists for every function
missFile = "/home/ctonline/MachineLearning/FEL/NotHits.hdf5"

background = backgroundReader(missFile)

hitFile = "/home/ctonline/MachineLearning/FEL/Hits.hdf5"
imageStackOfHits, listOfHits = H5reader(hitFile)

imageStackOfMisses, listOfMisses = H5reader(missFile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  parser.add_argument(
    #      '--image_dir',
    #      type=str,
    #      default='',
    #      help='Path to folders of labeled images.'
    #  )
    parser.add_argument(
      '--output_graph',
      type=str,
      default='/home/ctonline/MachineLearning/outputGraph3.pb',
      help='Where to save the trained graph.'
    )
    parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
      '--output_labels',
      type=str,
      default='/home/ctonline/MachineLearning/output_labels3.txt',
      help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
    )
    parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
    )
    parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
    )
    parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
    )
    parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
    )
    parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
      '--flip_left_right',
      default=True,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
    )
    parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
      Which TensorFlow Hub module to use.
      See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
      for some publicly available ones.\
      """)
    parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='',
      help='Where to save the exported graph.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

