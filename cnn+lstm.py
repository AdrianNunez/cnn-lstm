import os
import numpy as np
import cv2
import glob
import gc
import h5py
import time
import sys
import psutil
import shutil

import tensorflow as tf
from keras import backend as K
from keras import backend as K
from keras.models import Sequential, Layer
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.applications import VGG16
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, Merge, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from scipy.misc import imresize
from scipy.ndimage import imread
from collections import Counter
import scipy.io as sio

class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
        
def load_gaze_plus(batch_size, timestep):
    data_folder = '/home/anunez/gaze+/gaze_plus/'
    num_classes = 44
    image_shape = (224,224)
    crop_shape = (256,256)
    image_limit = 20000
    image_limit = image_limit - (image_limit%batch_size)
    
    mean = sio.loadmat('VGG_mean.mat')['image_mean']
    
    classes = dict()
    actors = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    actors.sort()
    label = 0
    for actor in actors:
        actions = [f for f in os.listdir(data_folder + actor) if os.path.isdir(os.path.join(data_folder + actor, f))] 
        actions.sort()
        for action in actions: 
            if action not in classes:
                classes[action] = label
                label += 1

    train_data, train_labels = [], []
    test_data, test_labels = [], []
    for actor in actors:
        actions = [f for f in os.listdir(data_folder + actor) if os.path.isdir(os.path.join(data_folder + actor, f))] 
        actions.sort()
        for action in actions:
            instances = [f for f in os.listdir(data_folder + actor + '/' + action) if os.path.isdir(os.path.join(data_folder + actor + '/' + action + '/', f))] 
            instances.sort()    
            for instance in instances[:int(len(instances)*0.8)]:
                train_data.append(data_folder + actor + '/' + action + '/' + instance)
                train_labels.append(classes[action])
            for instance in instances[int(len(instances)*0.8):]:
                test_data.append(data_folder + actor + '/' + action + '/' + instance)
                test_labels.append(classes[action])
    
    cnt_train = Counter()
    folders_in_class_train = dict()
    class_of_folder_train = dict()
    cnt_test = Counter()
    folders_in_class_test = dict()
    class_of_folder_test = dict()

    nb_images_train = 0
    for folder, label in zip(train_data, train_labels):
        x_frames = glob.glob(folder + '/*')
        nb_images = len(x_frames)
	nb_images_train += nb_images
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]
        if not folders_in_class_train.has_key(class_name):  
            folders_in_class_train[class_name] = []
        folders_in_class_train[class_name].append(folder)
        if not cnt_train.has_key(class_name):  
            cnt_train[class_name] = nb_images
        else:
            cnt_train[class_name] += nb_images
        class_of_folder_train[folder] = label

    nb_images_test = 0
    for folder, label in zip(test_data, test_labels):
        x_frames = glob.glob(folder + '/*')
        nb_images = len(x_frames)
        nb_images_test += nb_images
        temp = folder[:folder.rfind('/')]
        class_name = temp[temp.rfind('/')+1:]
        if not folders_in_class_test.has_key(class_name):  
            folders_in_class_test[class_name] = []
        folders_in_class_test[class_name].append(folder)
        cnt_test[folder] = nb_images
        if not cnt_test.has_key(class_name):  
            cnt_test[class_name] = nb_images
        else:
            cnt_test[class_name] += nb_images
        class_of_folder_test[folder] = label
        
    #max_class = cnt_train.most_common()[0][0]
    #nb_images_train = cnt_train[max_class]*num_classes
    
    print('Start loading dataset')
    while True:     
        if image_limit < nb_images_train:
            #stack_limit_train = nb_total_stacks_train - (nb_total_stacks_train%batch_size) + batch_size
            image_limit_train = nb_images_train
        else:
            image_limit_train = image_limit
            
        print('Number of images for train and test: {} - {}'.format(nb_images_train, nb_images_test))
        stack_size = 0
        counter = 0
        perm = np.random.permutation(num_classes)
        batches = []
        batch_labels = []
        num_batches = 0
        for p, idx in zip(np.asarray(folders_in_class_train.keys())[list(perm)], range(len(folders_in_class_train.keys()))):
            folders = folders_in_class_train[p]
            images_of_class_train = []
            labels_of_class_train = []
            counter += 1
            for element in folders:
                frames = glob.glob(element + '/*')
		j = 0
		sequence = np.zeros((timesteps, W, H, C))
		frames.sort()
		rest = len(frames) % timesteps
                for frame in frames[:len(frames)-rest]:    
                    img = cv2.imread(frame) 
                    sequence[j, ...] = img-mean
		    j += 1
		    if j == timestep:
                        j = 0
			images_of_class_train.append(sequence) 
			sequence = np.zeros((timesteps, W, H, C))
                        labels_of_class_train.append(to_categorical(class_of_folder_train[element], num_classes))
		if rest > 0:
			sequence = np.zeros((timesteps, W, H, C))
			frames.sort()
			j = 0
			for frame in frames[-timesteps:]:
				img = cv2.imread(frame) 
				sequence[j, ...] = img-mean
				j += 1
			images_of_class_train.append(sequence) 
			labels_of_class_train.append(to_categorical(class_of_folder_train[element], num_classes))
            #if len(images_of_class_train) < cnt_train[max_class]:
            #   diff = cnt_train[max_class]-len(images_of_class_train)
            #   if diff > 0:
            #       for elem in list(np.random.choice(range(len(images_of_class_train)), diff)):
            #           images_of_class_train.append(images_of_class_train[elem])
            #           labels_of_class_train.append(labels_of_class_train[elem])
            for elem in images_of_class_train:
                batches.append(elem)
            for elem in labels_of_class_train:
                batch_labels.append(elem)
            #added_images = len(images_of_class_train)
            #stack_size += added_images
            del images_of_class_train, labels_of_class_train
            gc.collect()
	stack_size = len(batches)
        num_batches = int(stack_size/batch_size)
        print('Train => total batches: {}, total images: {}'.format(num_batches, num_batches*batch_size*timesteps))
        for b in range(int(stack_size/batch_size)):
            yield np.asarray(batches[b*batch_size:(b+1)*batch_size]), np.asarray(batch_labels[b*batch_size:(b+1)*batch_size]), num_batches, int(nb_images_train/batch_size)
        del batches, batch_labels
        gc.collect()

	print('Testing')
        # TEST DATA
        stack_size = 0
        counter = 0
        perm = np.random.permutation(num_classes)
        batches = []
        batch_labels = []
        num_batches = 0

        for p, idx in zip(np.asarray(folders_in_class_test.keys())[list(perm)], range(len(folders_in_class_test.keys()))):
            folders = folders_in_class_test[p]
            images_of_class_test = []
            labels_of_class_test = []
            counter += 1
            for element in folders:
                frames = glob.glob(element + '/*')
		j = 0
		sequence = np.zeros((timesteps, W, H, C))
		frames.sort()
                rest = len(frames) % timesteps
                for frame in frames[:len(frames)-rest]:        
                    img = cv2.imread(frame) 
                    sequence[j, ...] = img-mean
		    j += 1
		    if j == timestep:
                        j = 0
                        images_of_class_test.append(sequence)
			sequence = np.zeros((timesteps, W, H, C))
                        labels_of_class_test.append(to_categorical(class_of_folder_test[element], num_classes))
		if rest > 0:
			sequence = np.zeros((timesteps, W, H, C))
			frames.sort()
			j = 0
			for frame in frames[-timesteps:]:
				img = cv2.imread(frame) 
				sequence[j, ...] = img-mean
				j += 1
			images_of_class_test.append(sequence) 
			labels_of_class_test.append(to_categorical(class_of_folder_test[element], num_classes))

            for elem in images_of_class_test:
                batches.append(elem)
            for elem in labels_of_class_test:
                batch_labels.append(elem)
            #added_images = len(images_of_class_test)
            #stack_size += added_images
            del images_of_class_test, labels_of_class_test
            gc.collect()
 	stack_size = len(batches)
        num_batches = int(stack_size/batch_size)
        print('Test => total batches: {}, total images: {}'.format(num_batches, num_batches*batch_size*timesteps))
        for b in range(int(stack_size/batch_size)):
            yield np.asarray(batches[b*batch_size:(b+1)*batch_size]), np.asarray(batch_labels[b*batch_size:(b+1)*batch_size]), num_batches, int(nb_images_test/batch_size)
        del batches, batch_labels
        gc.collect()

def show_RAM():
    values = psutil.virtual_memory()
    used = values.used / (1024*1024)
    active = values.active / (1024*1024)
    print('RAM: {}MB, {}MB'.format(used, active))

def relu(x):
    return tf.nn.relu(x)
    
def lrn(previous_layer, name):
    return tf.nn.local_response_normalization(previous_layer, bias=1.0, depth_radius=5, alpha=0.0005, beta=0.75, name=name)

# Variables    
weights_file = 'weights_cnnm2048_imagenet.npy'
checkpoint_folder = '/home/anunez/gaze+/checkpoints/'
best_model_file = 'best'
logs_path = '/home/anunez/gaze+/logs/'
plots_folder = '/home/anunez/gaze+/plots/'
num_classes = 44
L = 10
epochs = 50
batch_size = 1
# Oxford: 10e-2 => (50K) 10e-3 => (70K) 10e-5 => 80K stop
learning_rate = 0.0005
keep_probability = 0.5
weight_decay_rate = 0.0005
image_shape = (224,224,3)
num_units = 256
timesteps = 40
W = 224
H = 224
C = 3
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load the weights of Caffe for the CNN-M-2048 pre-trained in Imagenet
data = np.load(weights_file)
data = data[()]
x = tf.placeholder(tf.float32, (None, timesteps) + image_shape)
keep_prob = tf.placeholder(tf.float32)

weight_decay = tf.constant(weight_decay_rate, dtype=tf.float32)

# Create variables that take as an initializer the value of the weights of Caffe. Values cannot be passed directly, as they would be treated as constants.
conv1_weights = tf.get_variable('conv1_weights', initializer=data['conv1']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv1_biases = tf.get_variable('conv1_biases', initializer=data['conv1']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv2_weights = tf.get_variable('conv2_weights', initializer=data['conv2']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv2_biases = tf.get_variable('conv2_biases', initializer=data['conv2']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv3_weights = tf.get_variable('conv3_weights', initializer=data['conv3']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv3_biases = tf.get_variable('conv3_biases', initializer=data['conv3']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv4_weights = tf.get_variable('conv4_weights', initializer=data['conv4']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv4_biases = tf.get_variable('conv4_biases', initializer=data['conv4']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv5_weights = tf.get_variable('conv5_weights', initializer=data['conv5']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
conv5_biases = tf.get_variable('conv5_biases', initializer=data['conv5']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
flatten_weights = tf.get_variable('fc6_weights', initializer=data['fc6']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
flatten_biases = tf.get_variable('fc6_biases', initializer=data['fc6']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc7_weights = tf.get_variable('fc7_weights', initializer=data['fc7']['weights'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc7_biases = tf.get_variable('fc7_biases', initializer=data['fc7']['biases'])#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
fc_weights = tf.Variable(tf.truncated_normal([num_units, num_classes], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
fc_biases = tf.Variable(tf.zeros([num_classes]))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

#fc6_weights = tf.Variable(tf.truncated_normal([18432, 4096], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc6_biases = tf.Variable(tf.truncated_normal([4096], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc7_weights = tf.Variable(tf.truncated_normal([4096, 2048], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc7_biases = tf.Variable(tf.truncated_normal([2048], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc8_weights = tf.Variable(tf.truncated_normal([2048, num_classes], stddev=0.35))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
#fc8_biases = tf.Variable(tf.zeros([num_classes]))#, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

# LAYER 1
x_reshaped = tf.reshape(x, [-1, W, H, C])
conv_kernel_1 = tf.nn.conv2d(x_reshaped, conv1_weights, strides=[1,2,2,1], padding='VALID', name='conv1')
bias_layer_1 = relu(tf.nn.bias_add(conv_kernel_1, conv1_biases))
normalized_layer_1 = lrn(bias_layer_1, 'norm1')
pooled_layer1 = tf.nn.max_pool(normalized_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

# LAYER 2
padded_layer_2 = tf.pad(pooled_layer1, [[0,0], [1,1], [1,1], [0,0]], "CONSTANT")
conv_kernel_2 = tf.nn.conv2d(padded_layer_2, conv2_weights, strides=[1,2,2,1], padding='SAME', name='conv2')
bias_layer_2 = relu(tf.nn.bias_add(conv_kernel_2, conv2_biases))
normalized_layer_2 = lrn(bias_layer_2, name='norm2')
pooled_layer2 = tf.nn.max_pool(normalized_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

# LAYER 3
padded_layer_3 = tf.pad(pooled_layer2, [[0,0], [1,1], [1,1], [0,0]], "CONSTANT")
conv_kernel_3 = tf.nn.conv2d(padded_layer_3, conv3_weights, strides=[1,1,1,1], padding='VALID', name='conv3')
bias_layer_3 = relu(tf.nn.bias_add(conv_kernel_3, conv3_biases))

# LAYER 4
padded_layer_4 = tf.pad(bias_layer_3, [[0,0], [1,1], [1,1], [0,0]], "CONSTANT")
conv_kernel_4 = tf.nn.conv2d(padded_layer_4, conv4_weights, strides=[1,1,1,1], padding='VALID', name='conv4')
bias_layer_4 = relu(tf.nn.bias_add(conv_kernel_4, conv4_biases))

# LAYER 5
padded_layer_5 = tf.pad(bias_layer_4, [[0,0], [1,1], [1,1], [0,0]], "CONSTANT")
conv_kernel_5 = tf.nn.conv2d(padded_layer_5, conv5_weights, strides=[1,1,1,1], padding='VALID', name='conv4')
bias_layer_5 = relu(tf.nn.bias_add(conv_kernel_5, conv5_biases))
pooled_layer5 = tf.nn.max_pool(bias_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

flattened = tf.contrib.layers.flatten(pooled_layer5)
flattened = relu(tf.nn.bias_add(tf.matmul(flattened, flatten_weights), flatten_biases))
flattened = tf.reshape(flattened, [batch_size, timesteps, 4096])
flattened = tf.transpose(flattened, [1,0,2])
#flattened = tf.split(flattened, timesteps, 0)
#flattened = tf.unstack(tf.reshape(flattened, [batch_size, timesteps, 2359296]))
# LSTM ===========================
lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1)
lstm_layer = tf.nn.rnn_cell.DropoutWrapper(lstm_layer, output_keep_prob=keep_prob)
outputs, states = tf.nn.dynamic_rnn(cell=lstm_layer, inputs=flattened, dtype=tf.float32, time_major=True, parallel_iterations=2)
#outputs = tf.reshape(outputs, [-1, num_units])
# LSTM ===========================

#fc_layer6 = relu(tf.nn.bias_add(tf.matmul(outputs[-1], fc6_weights), fc6_biases))
#dropout_1 = tf.nn.dropout(fc_layer6, keep_prob)
#fc_layer7 = relu(tf.nn.bias_add(tf.matmul(dropout_1, fc7_weights), fc7_biases))
#dropout_2 = tf.nn.dropout(fc_layer7, keep_prob)
fc_layer = tf.nn.bias_add(tf.matmul(outputs[-1], fc_weights), fc_biases) #uses last output
preds = tf.nn.softmax(fc_layer) # Not a good practice to compute softmax by ourselves

y = tf.placeholder(tf.float32, shape=(None, num_classes))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc_layer))
#opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
optimizer = opt.minimize(loss_op)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(preds,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.summary.scalar("cost", loss_op)
tf.summary.scalar("accuracy", accuracy)

# Initialize all variables
init_op = tf.global_variables_initializer()

session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
#session_config.gpu_options.per_process_gpu_memory_fraction = 0.90
#session_config.gpu_options.allow_growth=True 


if os.path.exists(logs_path):
   shutil.rmtree(logs_path)
os.makedirs(logs_path)

summary_op = tf.summary.merge_all()
show_RAM()
saver = tf.train.Saver()
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
summary_writer = tf.summary.FileWriter(logs_path)
    
with tf.Session(config=session_config) as sess:
    start = time.time()
    sess.run(init_op)
    summary_writer.add_graph(sess.graph)
    epoch_start = time.time()  
    best_accuracy = 0.
    best_acc_epoch = -1
    loss_train, acc_train = [], []
    loss_test, acc_test = [], []
    stack_generator = load_gaze_plus(batch_size, timesteps)
    for e in range(epochs):    
         epoch_start = time.time()
         i = 0
         avg_cost = 0
         avg_acc = 0
         time_to_load_train = 0
         nb_batches  = 0
	 print('Training')
         j = 0
	 train_start = time.time()
         while True:
                 temp = time.time()
                 batch_x, batch_y, num_batches, total_images = stack_generator.next()
		 
                 #print('Time to load batch: {}s'.format(time.time()-temp))
                 time_to_load_train += time.time()-temp
                 #print('Generate batch: {} seconds'.format(time.time()-temp))
                 _, training_loss, summary, training_accuracy = sess.run([optimizer, loss_op, summary_op, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability, K.learning_phase(): 1})
                 
                 #if i % 10 == 0:
                 #    print("   interepoch - Loss: {0:.2f}, Accuracy: {0:.2f}".format(training_loss, training_accuracy))
                 #summary_writer.add_summary(summary, nb_batches+j)
                 avg_cost += training_loss / num_batches
                 avg_acc += training_accuracy / num_batches
                 j += 1
                 #sys.stdout.write("\r[{}>{}{} ".format("="*j, ' '*(num_batches-j-1), ']'))
                 #sys.stdout.flush()
                 if j >= num_batches:
                     break 
         loss_train.append(avg_cost)
         acc_train.append(avg_acc)
	 train_end = time.time()
         i = 0
         test_loss = 0.
         test_acc = 0.
         time_to_load_test = 0
         nb_batches = 0
         # TEST
	 print('Validation')
         j = 0
	 test_start = time.time()
         while True:
                 temp = time.time()
                 batch_x, batch_y, num_batches, total_batches = stack_generator.next()
                 #cv2.imwrite('image{}.jpg'.format(j), batch_x[0,:,:,0], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                 #print('Time to load batch: {}s'.format(time.time()-temp))
                 time_to_load_test += time.time()-temp
                 #print('Generate batch: {} seconds'.format(time.time()-temp))
                 testing_loss, summary, testing_accuracy = sess.run([loss_op, summary_op, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability, K.learning_phase(): 0})

                 #if i % 10 == 0:
                 #    print("   interepoch - Loss: {0:.2f}, Accuracy: {0:.2f}".format(training_loss, training_accuracy))
                 summary_writer.add_summary(summary, i)
                 test_loss += testing_loss / num_batches
                 test_acc += testing_accuracy / num_batches
                 j += 1
                 #sys.stdout.write("\r[{}>{}{}".format("="*j, ' '*(num_batches-j-1), ']'))
                 #sys.stdout.flush()
                 if j >= num_batches:
                     break   
	 loss_test.append(test_loss)
         acc_test.append(test_acc)
	 test_end = time.time()
         print('='*20)
         print("Epoch {}".format(e))
	 print('Max GPU in use: ', sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
         print("TRAIN: Loss: {:.3f}, Accuracy: {:.3f}, ETA: {}, data load time: {}".format(avg_cost, avg_acc, train_end-train_start, time_to_load_train))
         print("TEST: Loss: {:.3f}, Accuracy: {:.3f}, ETA: {}, data load time: {}".format(test_loss, test_acc, test_end-test_start, time_to_load_test))
         print('ETA total: {}'.format(time.time()-epoch_start))
         save_path = saver.save(sess, checkpoint_folder + 'epoch_{}-loss_{}-acc_{}'.format(e, test_loss, test_acc))
         if test_acc > best_accuracy:
             path = checkpoint_folder + best_model_file + '_{:.2f}-prev_{:.2f}'.format(test_acc, best_accuracy)
             print('Best result so far - saving model in {}'.format(path))
             save_path = saver.save(sess, path)
             best_accuracy = test_acc
	     best_acc_epoch = e
         print('='*20)
    
    plt.plot(range(epochs), acc_train)
    plt.savefig(plots_folder + 'train_acc.png')
    plt.gcf().clear()
    
    plt.plot(range(epochs), loss_train)
    plt.savefig(plots_folder + 'train_loss.png')
    plt.gcf().clear()
    
    plt.plot(range(epochs), acc_test)
    plt.savefig(plots_folder + 'test_acc.png')
    plt.gcf().clear()
    
    plt.plot(range(epochs), loss_test)
    plt.savefig(plots_folder + 'test_loss.png')
    plt.gcf().clear()
    
    print('Best accuracy: {} in epoch {}'.format(best_accuracy, best_acc_epoch))
    
    summary_writer.close()
