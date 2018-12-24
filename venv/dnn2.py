import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
#import data
from cnn_utils import random_mini_batches

#读取trani_data和train_label
X_train_orig=np.loadtxt('train_data2.csv',dtype=int,delimiter=',')
X_train_orig=np.reshape(X_train_orig,(1420,64,64,3))
Y_train_orig=np.loadtxt('train_label2.csv',dtype=int, delimiter=',')


X_test_orig=np.loadtxt('test_data2.csv',dtype=int,delimiter=',')
X_test_orig=np.reshape(X_test_orig,(300,64,64,3))
Y_test_orig=np.loadtxt('test_label2.csv',dtype=int, delimiter=',')

#显示一张图
# index = 6
# plt.imshow(X_test_orig[0])
# plt.show()
# plt.imshow(X_test_orig[29])
# plt.show()
# plt.imshow(X_test_orig[30])
# plt.show()
# plt.imshow(X_test_orig[59])
# plt.show()
# print(Y_test_orig[6])
# print(Y_test_orig[0])
# print(Y_test_orig[29])
# print(Y_test_orig[30])


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


#将数据集变成需要的格式
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 10).T
Y_test = convert_to_one_hot(Y_test_orig, 10).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))




def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(name='X', shape=(None, n_H0, n_W0, n_C0), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y

#初始化卷积参数，设置两层卷积
def initialize_parameters():

    with tf.name_scope("parameters"):
        tf.set_random_seed(1)
        W1 = tf.get_variable('W1', [4, 4, 3, 8], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable('W2', [2, 2, 8, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))


    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']

    with tf.name_scope("layer1"):
        Z1 = tf.nn.conv2d(input=X, filter=W1, strides=(1, 1, 1, 1), padding='SAME')
        tf.summary.scalar('layer1_Z1', Z1)

        A1 = tf.nn.relu(Z1)
        tf.summary.scalar('layer1_A1', A1)

        P1 = tf.nn.max_pool(value=A1, ksize=(1, 8, 8, 1), strides=(1, 8, 8, 1), padding='SAME')
        tf.summary.scalar('layer1_P1', P1)
    # CONV2D: filters W2, stride 1, padding 'SAME'

    with tf.name_scope("layer2"):
        Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=(1, 1, 1, 1), padding='SAME')
        tf.summary.scalar('layer2_Z2', Z2)
    # RELU
        A2 = tf.nn.relu(Z2)
        tf.summary.scalar('layer2_A2', A2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.nn.max_pool(value=A2, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')
        tf.summary.scalar('layer2_P2', P2)
    # FLATTEN
        P2 = tf.contrib.layers.flatten(inputs=P2)
        F2 = tf.nn.dropout(P2, keep_prob=0.5)
        tf.summary.scalar('layer2_dropout_F2', F2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"

    with tf.name_scope("output"):
        Z3 = tf.contrib.layers.fully_connected(F2, 10, activation_fn=None)
        tf.summary.scalar('output', Z3)


    return Z3


def compute_cost(Z3, Y):
    with tf.name_scope("loss"):

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
        tf.summary.scalar('cross_entro', cost)

    return cost


def model(X_train, Y_train,  X_test, Y_test, learning_rate=0.003,
          num_epochs=50, minibatch_size=64, print_cost=True):


    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    with tf.name_scope("input"):
        X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
        tf.summary.scalar('input_X', X)
        tf.summary.scalar('input_Y', Y)
    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #tf.summary.scalar('Adam', optimizer)
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter("/anaconda3/dnn/con2", tf.get_default_graph())
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            iiiii = 0
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                iiiii += 1
                if iiiii ==10:
                    run_options = tf.RunOptions( trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    opti, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y}, \
                                               options=run_options, run_metadata=run_metadata)

                    writer.add_run_metadata(run_metadata, str(epoch))
                else:

                    opti, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    #tf.summary.scalar('opti', opti)
                tf.summary.scalar('temp_cost', temp_cost)
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True :#and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True :#and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate = 0.003" )
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        #print("Train Accuracy: 0.99647886")
        #print("Test Accuracy: 0.99")
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        #writer = tf.summary.FileWriter("/anaconda3/dnn/con1", tf.get_default_graph())
        writer.close()
        return train_accuracy, test_accuracy, parameters




_, _, parameters = model(X_train, Y_train,  X_test, Y_test, )


# def forward_propagation_for_picture(X, parameters):
#     """
#     Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
#
#     Arguments:
#     X -- input dataset placeholder, of shape (input size, number of examples)
#     parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
#                   the shapes are given in initialize_parameters
#
#     Returns:
#     Z3 -- the output of the last LINEAR unit
#     """
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     # Retrieve the parameters from the dictionary "parameters"
#     W1 = parameters['W1']
#     b1 = parameters['b1']
#     W2 = parameters['W2']
#     b2 = parameters['b2']
#     W3 = parameters['W3']
#     b3 = parameters['b3']
#
#     # Numpy Equivalents:
#     Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
#     A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
#     a_1 = A1.eval()
#     Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
#     A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
#     a_2 = A2.eval()
#     Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
#
#     return a_1, a_2
#
# p1,p2 =forward_propagation_for_picture(X_train_orig[6], parameters)
# print(p_1.shape)
# print(p_2.shape)
# p1=np.concatenate((p1,p2),0)
# print(p_1.shape)
# #np.savetxt('p1p2.csv',train_data,fmt='%d',delimiter=',')
# # print(parameters)
# # index = 6
# # plt.imshow(X_train_orig[6])
# # plt.show()

