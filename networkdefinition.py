# импорт необходимых пакетов
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K


class NetworkDefinition:
	@staticmethod
	def build(width, height, depth, classes):

		#https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
		# инициализация модели вместе с входной формой,
		# чтобы она была c «каналами в конце» и их размером
		# (в некоторых случаях влияет на производительность)
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# если же используются "каналы в начале", нужно обновить входную форму
		# и измерение каналов
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# первый слой
		# CONV - свертка,
		# RELU - функция активации, https://towardsdatascience.com/convolution-neural-networks-a-beginners-guide-implementing-a-mnist-hand-written-digit-8aa60330d022
		# POOL - объединение слоев https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/
		# CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# второй слой
		# CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# первый и единственный набор
		# FC - полносвязный слой
		# https://www.researchgate.net/figure/CNN-represents-convolutional-neural-network-layers-and-FC-represents-fully-connected_fig1_334695087
		# http://indiantechwarrior.com/fully-connected-layers-in-convolutional-neural-networks/
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# https://cs231n.github.io/linear-classify/#:~:text=The%20Softmax%20classifier%20uses%20the,entropy%20loss%20can%20be%20applied.
		# https://konstantinklepikov.github.io/2019/06/27/cs2131n-sofrmax.html#:~:text=Softmax%2D%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%82%D0%BE%D1%80%20%D1%81%D0%B2%D0%BE%D0%B4%D0%B8%D1%82%20%D0%BA%20%D0%BC%D0%B8%D0%BD%D0%B8%D0%BC%D1%83%D0%BC%D1%83,%D0%B0%20%D0%B2%D0%B5%D1%80%D0%BE%D1%8F%D1%82%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%BE%D1%81%D1%82%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D1%85%20%D1%80%D0%B0%D0%B2%D0%BD%D1%8B%200.
		# Softmax-классификатор
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# возврат построенной сетевой архитектуры
		return model

