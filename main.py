# Python 3.5.4: Anaconda
# Requires scipy, PIL, numpy, tensorflow, glob
# coding: utf-8

# In[ ]:


###TEAM###
#Sunil Umasankar (UBITName = suniluma, personNumber = 50249002) 
#Prajna Bhandary (UBITName = prajnaga, personNumber = 50244304)
#Abhishek Subramaniam (UBITName = a45, personNumber = 50244979)


# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob


# In[2]:


#Importing MNIST from tensorflow
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#Importing MNIST from sklearn for logistic regression
from sklearn.datasets import fetch_mldata
mnist_sklearn = fetch_mldata('MNIST original')


# In[3]:


#USPS Importing

from PIL import Image

def resizer(path):
    return np.array(Image.open(path).convert('L').resize((28,28), Image.ANTIALIAS)).ravel().tolist()

def images_from_dirs(dirs):
    train_img = 255 - np.array([resizer(i) for i in glob.glob(dirs)])
    return train_img

train_img = images_from_dirs(".\\USPS_data\\Numerals\\*\\*.png")
test_img = images_from_dirs(".\\USPS_data\\Test\\*.png")

training_Y  = np.array([float(int(i/2000)) for i in range(20000)])
testing_Y = np.array([float(int(i/150)) for i in range(1500)])[::-1]


# In[4]:


#One hot representation of test data for Neural Networks
YY = np.array([(int(i/150)) for i in range(1500)])[::-1]
one_hot_YY = np.zeros((YY.size, YY.max()+1))
one_hot_YY[np.arange(YY.size),YY] = 1


# In[5]:


YY2 = np.array([(int(i/2000)) for i in range(20000)])
training_one_hot_YY = np.zeros((YY2.size, YY2.max()+1))
training_one_hot_YY[np.arange(YY2.size),YY2] = 1


# # Logistic Regression

# In[6]:


#Random split of data
from sklearn.model_selection import train_test_split

#MNIST
train_X, test_X, train_Y, test_Y = train_test_split( mnist_sklearn.data, mnist_sklearn.target, test_size=1/7.0)

#USPS Numerals
usps_train_X, usps_test_X, usps_train_Y, usps_test_Y = train_test_split(train_img,training_Y, test_size=1/7.0)


# In[7]:


#Hyperparameters
MAX_ITER = 100
LEARNING_RATE = 1.0


# In[8]:


def train_logistic_regression(train_X,train_Y,max_iter,learn_rate):
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(solver = 'lbfgs', multi_class="multinomial", max_iter=max_iter, C=learn_rate)
    logisticRegr.fit(train_X, train_Y)
    return logisticRegr

def accuracy_logistic_regression(trained_model, test_X, test_Y):
    predictions = trained_model.predict(test_X)
    score = trained_model.score(test_X, test_Y)
    return score


# In[9]:


trained_model = train_logistic_regression(train_X,train_Y,MAX_ITER,LEARNING_RATE)
print("Accuracy on MNIST Testing Data: " + str(accuracy_logistic_regression(trained_model,test_X,test_Y)))
print("Accuracy on USPS Testing Data: " + str(accuracy_logistic_regression(trained_model,test_img,testing_Y)))


# In[10]:


print("Training on USPS Data and Testing")
trained_model = train_logistic_regression(usps_train_X,usps_train_Y,MAX_ITER,LEARNING_RATE)
print("Accuracy on Training Data: " + str(accuracy_logistic_regression(trained_model,usps_test_X,usps_test_Y)))
print("Accuracy on Testing Data: " + str(accuracy_logistic_regression(trained_model,test_img,testing_Y)))


# # Single Hidden Layer NN

# In[11]:


#Hyperparameters
MAX_ITER = 10000
LEARNING_RATE = 0.2
BATCH_SIZE = 50


# In[12]:


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,  W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(MAX_ITER):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy on MNIST")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("Accuracy on USPS")
print(sess.run(accuracy, feed_dict={x: test_img, y_: one_hot_YY}))


# # CNN

# In[13]:


#Hyperparameters
MAX_ITER = 1000
BATCH_SIZE = 50
LEARNING_RATE = 1e-4


# In[14]:


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


# In[15]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')


# In[16]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[17]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[18]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[19]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[20]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# In[21]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(MAX_ITER):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('Step %d, Training Accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    print('MNIST Test Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('USPS Numerals Accuracy %g' % accuracy.eval(feed_dict={x: train_img, y_: training_one_hot_YY, keep_prob: 1.0}))
    print('USPS Test Accuracy %g' % accuracy.eval(feed_dict={x: test_img, y_: one_hot_YY, keep_prob: 1.0}))

