import numpy as np


def load_img(fpath):
	with open(fpath, "rb") as f:
		# バッファからnumpy行列を作成する
		# 16進数データで見たほうがわかりやすい
		# 先頭16バイトは説明データなので省略
		data = np.frombuffer(f.read(), np.uint8, offset=16)
	# 28x28=784
	data = data.reshape(-1, 784)
	return data


def to_one_hot(label):
    T = np.zeros((label.size, 10))
    for i in range(label.size):
        T[i][label[i]] = 1
    return T


def load_label(fpath, onehot=True):
	with open(fpath, "rb") as f:
		# バッファからnumpy行列を作成する
		# 16進数データで見たほうがわかりやすい
		# 先頭8バイトは説明データなので省略
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	if onehot:
		labels = to_one_hot(labels)
	return labels