# установка matplotlib для записи изображений в фоне
import matplotlib

# импорт необходимых пакетов
from networkdefinition import NetworkDefinition

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizer_v2 import adam
from keras.utils.np_utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

matplotlib.use("Agg")

#python train.py --dataset dataset --model liveness.model --le le.pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# построение парсинга аргументов и их параметров
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="путь к входному набору данных")
ap.add_argument("-m", "--model", type=str, required=True, help="путь к тренированной модели")
ap.add_argument("-l", "--le", type=str, required=True, help="путь к енкодеру классов")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="путь сохранения графика потерь/точности")
args = vars(ap.parse_args())

# инициализация начальной скорости обучения, размера партии и числа эпох для тренировки
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# получение списка изображений из каталога набора данных и их
# инициализация в набор данных изображений и классов
print("[INFO] загрузка изображений...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# цикл перебора всех изображений
for imagePath in imagePaths:
	# извлечение имени класса с названия изображения, загрузка и
	# изменение размера изображения к формату 32x32 пикселя игнорируя соотношение сторон
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
	# обновление списков данных и меток (классов)
	data.append(image)
	labels.append(label)

# конвертация данных в массив NumPy, их предобработка изменением
# интенсивности всех пикселей в диапазоне [0, 1]
data = np.array(data, dtype="float") / 255.0

# кодировать метки (которые в настоящее время являются строками) как целые числа
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# разделение данных на обучающие и тестовые,
# 75% данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.25, random_state=12)

# построение генератора обучающих изображений для прироста данных
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.25, width_shift_range=0.25, height_shift_range=0.2,
						 shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# инициализировать оптимизатор и модель
print("[INFO] компиляция модели...")
opt = adam.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = NetworkDefinition.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# обучение сети
print("[INFO] старт обучения {} эпох...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
			  steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# оценка сети
print("[INFO] оценка сети...")
predictions = model.predict(x=testX, batch_size=12)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# запись модели на диск
print("[INFO] сериализация модели '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

# сохранить кодировщик меток на диск le (label encoder)
file = open(args["le"], "wb")
file.write(pickle.dumps(le))
file.close()

# построить график обучающих потерь и точности
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Потери и точность на наборе данных")
plt.xlabel("Эпохи #")
plt.ylabel("Потеря/Точность")
plt.legend(loc="center right")
plt.savefig(args["plot"])
