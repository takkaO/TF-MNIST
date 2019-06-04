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

	#initializer = tf.contrib.layers.variance_scaling_initializer()
	hidden_size = 50
	input_size = 784
	output_size = 10
	train_size = train_img.shape[0]
	test_size = test_img.shape[0]
	batch_size = 100

	x = tf.placeholder("float", [None, input_size])
	y = tf.placeholder("float", [None, output_size])

	"""
	W1 = tf.Variable(initializer([input_size, hidden_size]))
	b1 = tf.Variable(tf.zeros([hidden_size]))
	W2 = tf.Variable(initializer([hidden_size, output_size]))
	b2 = tf.Variable(tf.zeros([output_size]))

	y1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
	logits = tf.add(tf.matmul(y1, W2), b2)
	"""
	#predict = tf.nn.softmax(logits)

	# TODO:廃止予定なので確認
	X = tf.reshape(x, shape=[-1, 28, 28, 1])
	conv1 = tf.layers.conv2d(X, 32, 3, activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
	flat = tf.layers.flatten(pool1)
	fc1 = tf.layers.dense(flat, hidden_size)
	logits = tf.layers.dense(fc1, output_size)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		logits=logits,
		labels=y
	)


	opt = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)  # SGD
	#opt = tf.train.AdamOptimizer().minimize(loss)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	init = tf.global_variables_initializer()

	print("Train start")
	epochs = 10000
	with tf.Session() as sess:
		sess.run(init)
		for step in range(epochs):
			batch_mask = np.random.choice(train_size, batch_size)
			batch_x = train_img[batch_mask]
			batch_t = train_label[batch_mask]

			sess.run(opt, feed_dict={x: batch_x, y: batch_t})

			if step % (1000) == 0:
				print(step, "/", epochs, "Complete! ")
				batch_mask = np.random.choice(train_size, batch_size)
				batch_x = train_img[batch_mask]
				batch_t = train_label[batch_mask]
				res = sess.run(accurancy, feed_dict={x: batch_x, y: batch_t})
				print("train accurancy: ", "{:.2f}".format(res*100.0), " %")
				batch_mask = np.random.choice(test_size, batch_size)
				batch_x = test_img[batch_mask]
				batch_t = test_label[batch_mask]
				res = sess.run(accurancy, feed_dict={x: batch_x, y: batch_t})
				print("test  accurancy: ", "{:.2f}".format(res*100.0), " %")
				
		print(epochs, "/", epochs, "Complete! ")
		batch_mask = np.random.choice(train_size, batch_size)
		batch_x = train_img[batch_mask]
		batch_t = train_label[batch_mask]
		res = sess.run(accurancy, feed_dict={x: batch_x, y: batch_t})
		print("train accurancy: ", "{:.2f}".format(res*100.0), " %")
		batch_mask = np.random.choice(test_size, batch_size)
		batch_x = test_img[batch_mask]
		batch_t = test_label[batch_mask]
		res = sess.run(accurancy, feed_dict={x: batch_x, y: batch_t})
		print("test  accurancy: ", "{:.2f}".format(res*100.0), " %")

		#saver = tf.train.Saver()
		#saver.save(sess, "./model/test_model")
	
	

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
