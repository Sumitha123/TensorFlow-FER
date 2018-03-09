import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import datapreprocessing

# Splitting data into training & validation to reduce overfitting
validation_size = 1709
validation_images = datapreprocessing.images[:validation_size]
validation_labels = datapreprocessing.labels[:validation_size]

train_images = datapreprocessing.images[validation_size:]
train_labels = datapreprocessing.labels[validation_size:]
print('Number of images in final training dataset : ', (len(train_images)))

# Initialization of weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    return tf.Variable(initial)

#Initialization of bias variables
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Convolution
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)

# Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# Creating placeholder for image data
x = tf.placeholder('float', shape=[None, datapreprocessing.image_pixels])
# Creating placeholder for labels
y_ = tf.placeholder('float', shape=[None, datapreprocessing.labels_count])

# First convolutional layer 64
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

# Reshaping the image data - (27000, 2304) --> (27000,48,48,1)
image = tf.reshape(x, [-1,datapreprocessing.image_width , datapreprocessing.image_height,1])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
#print (h_conv1.get_shape()) --> (27000,48,48,64)

h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) --> (27000,24,24,1)

h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
#print (h_conv2.get_shape()) --> (27000,24,24,128)

h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

h_pool2 = max_pool_2x2(h_norm2)

# Initialization of local layer weight
def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# Densely connected layer local 3
W_fc1 = local_weight_variable([12 * 12 * 128, 3072])
b_fc1 = local_bias_variable([3072])

# (27000, 12, 12, 128) --> (27000, 12 * 12 * 128)
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) --> (27000, 1024)

# Densely connected layer local 4
W_fc2 = local_weight_variable([3072, 1536])
b_fc2 = local_bias_variable([1536])

# (40000, 7, 7, 64) --> (40000, 3136)
h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])

h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)
#print (h_fc1.get_shape()) --> (40000, 1024)

# Dropout
keep_prob = tf.placeholder('float')
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Readout layer for deep net
W_fc3 = weight_variable([1536, datapreprocessing.labels_count])
b_fc3 = bias_variable([datapreprocessing.labels_count])

y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

#print (y.get_shape()) --> (40000, 10)
# Settings
LEARNING_RATE = 1e-4

# Cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# Optimization using Adam optimizer
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Prediction function
predict = tf.argmax(y,1)

# Setting training iteration to 4000
TRAINING_ITERATIONS = 4000

DROPOUT = 0.5
BATCH_SIZE = 50

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


# Serving data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # When all training data has been used, it is reordered randomly
    if index_in_epoch > num_examples:
        # Finished epoch
        epochs_completed += 1
        # Shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # Start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# Starting TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# Visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    # Get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # Checking progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_prob: 1.0})
        if (validation_size):
            validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                           y_: validation_labels[0:BATCH_SIZE],
                                                           keep_prob: 1.0})
            print('Training accuracy :  %.2f  Validation :  %.2f  for step %d' % (
            train_accuracy, validation_accuracy, i))

            validation_accuracies.append(validation_accuracy)

        else:
            print('Training accuracy :  %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        # Increasing display_step
        if i % (display_step * 10) == 0 and i and display_step < 100:
            display_step *= 10
    # Training on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})


if(validation_size):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    print('Validation accuracy :  %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-r', label='Training')
    plt.plot(x_range, validation_accuracies,'-c', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.0, ymin = 0.0)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
    plt.savefig("Training_val_accuracy.png")

saver = tf.train.Saver(tf.all_variables())
saver.save(sess, 'my-model1', global_step=0)






