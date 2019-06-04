import os
import numpy as np
import tensorflow as tf
from load_mnist import *

def printProgress(percent):
	print("\r{0} % Complete! ".format(percent), end='')

def main():
	train_img = load_img("./train-images.idx3-ubyte")
	train_label = load_label("./train-labels.idx1-ubyte", True)

	test_img = load_img("./t10k-images.idx3-ubyte")
	test_label = load_label("./t10k-labels.idx1-ubyte", True)

	initializer = tf.contrib.layers.variance_scaling_initializer()
	hidden_size = 50
	input_size = 784
	output_size = 10

	x = tf.placeholder("float", [None, input_size])
	y = tf.placeholder("float", [None, output_size])

	W1 = tf.Variable(initializer([input_size, hidden_size]))
	b1 = tf.Variable(tf.zeros([hidden_size]))
	W2 = tf.Variable(initializer([hidden_size, output_size]))
	b2 = tf.Variable(tf.zeros([output_size]))

	y1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
	logits = tf.add(tf.matmul(y1, W2), b2)
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "./model/test_model")
		print("Restore Model")

		print("Check Model")
		res = sess.run(accurancy, feed_dict={x: train_img, y: train_label})
		print("train accurancy: ", "{:.2f}".format(res*100), " %")
		res = sess.run(accurancy, feed_dict={x: test_img, y: test_label})
		print("test  accurancy: ", "{:.2f}".format(res*100), " %")
	

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
