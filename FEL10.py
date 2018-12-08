# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import numpy as np
import tensorflow as tf
import os
import h5py
#import scipy.misc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.ndimage import imread
import datetime
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import gc
from PIL import Image, ImageEnhance, ImageOps
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
           [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
           [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
                                                  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
           [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
                                                        0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
           [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
                                                  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
           [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
                                                        0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
           [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
                                                       0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429],
           [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
                                                        0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048,
                                                                                                 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
           [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
                                                        0.7607190476], [0.0383714286, 0.6742714286, 0.743552381],
           [0.0589714286, 0.6837571429, 0.7253857143],
           [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
           [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
                                                        0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048],
           [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
                                                        0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
           [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
                                                  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
           [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
                                                        0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905],
           [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
                                                  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
                                                        0.2886428571], [0.9738952381, 0.7313952381, 0.266647619],
           [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
                                                       0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
                                                  0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
           [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

class generator:
    def __init__(self, file):
        self.file = file
    
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf["train_img"]:
                yield im

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
    return graph


def read_tensor_from_image_file(snapshot,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    #Begin modified function
    image_reader = snapshot
    
    image_reader = tf.image.resize_images(image_reader, [299, 299])

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(image_reader, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    result = np.reshape(result, [-1, 299, 299, 3])
    
    #  result = tf.cast(result, tf.uint8)
#    result = tf.image.grayscale_to_rgb(
#                              result,
#                              name=None
#                              )
    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
        return label

##MAIN###########################################

def H5reader(filename) :
    color = pd.read_csv("~/MachineLearning/Parula.txt", header = None)
    color = np.array(color)
    Parula = LinearSegmentedColormap.from_list('parula', color)

    stack = [0]*100
    eIDStack = []
    for i in range(0, 100):
        
        
        file    = h5py.File(filename, 'r')
        dataset  = file['/data/data/']
        eID = file['/meta/external_id/']
        image = dataset[i]
#        print(np.max(image))
        image = turnOffK(image, 15)
        image = turnOffK(image, 16)
#        print(np.max(image))
        background = dataset[0]
        background = turnOffK(background, 15)
        background = turnOffK(background, 16)
   
        image = image - background

        # image[image > 5000] = 0
        image[0 > image] = 0
#        print(np.max(image))
        #print(image)
        image = rescale_linear(image, 0.0035713999999999997, 0.9988299999999999)
        
        image = Parula(image)
        image = np.uint8(image*255)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = image.rotate(270)
        #scale_value=scale1.get()
        #contrast_applied=contrast.enhance(scale_value)
        #  image = np.stack((image,)*3, -1)
        #stack.append(image)
        stack[i] = image
        eIDStack.append(eID[i])
        #        print(eID[i])
        #        eIDStack = eID[i]
    #Snapshotx, 1024, 1024, 3
    file.close()
    del dataset
    return stack, eIDStack
    del stack
    del eIDStack


if __name__ == "__main__":
    #np.savetxt("my_output_file.csv", labels, delimiter=",")
    #Defaults
   # file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
    #model_file = \
#"tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
 #   label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    #Parse from terminal We should not be using this
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    # parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()


input_layer = "Placeholder"
output_layer = "final_result"
graph = load_graph("/home/ctonline/MachineLearning/outputGraph3.pb")
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

#PATH TO LABELS HERE
labels = load_labels("/home/ctonline/MachineLearning/output_labels3.txt")

with open('TestRun001.csv', mode = 'w') as outFile:
 with tf.Session(graph=graph) as sess:
    k = 0
    wr = csv.writer(outFile, delimiter=',', lineterminator='\n')
    wr.writerow(["BatchID" , "Classification", "Results"])
    #loop over all the files in the directory. remove and replace with .h5 APIs

    for file_name in os.listdir("/home/ctonline/Desktop/Run 141 copy"):
        pastTime = datetime.datetime.now()
        if file_name.endswith(".h5"):
            k = k + 1
            print(k)
            file_name = "/home/ctonline/Desktop/Run 141 copy/" + file_name
            stackOfImages, stackOfBatchIDs = H5reader(file_name)
            for i in range(100):
                t = read_tensor_from_image_file(
                                                stackOfImages[i],
                                                input_height=input_height,
                                                input_width=input_width,
                                                input_mean=input_mean,
                                                input_std=input_std)

                results = sess.run(output_operation.outputs[0], {
                                       input_operation.outputs[0]: t
                                       })
                wr.writerow([stackOfBatchIDs[i] ,labels[0], results[0][0]])
                del t
                del results
            del stackOfImages
            del stackOfBatchIDs
            gc.collect()
            currentTime = datetime.datetime.now()
            print(currentTime - pastTime)




