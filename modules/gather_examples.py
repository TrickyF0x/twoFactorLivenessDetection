# импорт необходимых пакетов
import numpy as np
import argparse
import cv2
import os

#python modules/gather_examples.py --input ./videos/fake.mp4 --output ./dataset/fake --detector ./face_detector --skip 1
#python modules/gather_examples.py --input ./videos/real.mp4 --output ./dataset/real --detector ./face_detector --skip 2


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# построение парсинга аргументов и их параметров
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="путь к входному видео")
ap.add_argument("-o", "--output", type=str, required=True, help="путь к выходному каталогу обработанных лиц")
ap.add_argument("-d", "--detector", type=str, default="./face_detector",
                help="путь к детектору глубокого обучения OpenCV для обнаружения лиц")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="минимальная вероятность отфильтровать слабые обнаружения")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# кадров, которые нужно пропустить перед применением распознавания лиц")
args = vars(ap.parse_args())

# загрузка сериализованного детектора лиц с диска
print("[INFO] загрузка детектора лиц...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# открыть указатель на поток видеофайлов и инициализировать общее количество прочитанных и сохраненных кадров
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# цикл по кадрам из потока видеофайла
while True:
  print("[INFO] цикл запущен..")
  # сбор кадра из файла
  (grabbed, frame) = vs.read()

  # если кадр не схвачен, то достигнут конец потока, либо в нем нет лица
  if not grabbed:
    print("[INFO] достигнут конец потока...")
    break

  # рассчет общего количество кадров, прочитанных на данный момент
  read += 1

  # проверка, должен ли обрабатываться этот кадр
  if read % args["skip"] != 0:
    continue

  # получение размеров кадра и создание blob из кадра (подготовка изображения для нейросети)
  # https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                               (300, 300), (104.0, 177.0, 123.0))
  # передать blob через модель для получения обнаружений и прогнозов
  net.setInput(blob)
  detections = net.forward()
  # убедиться, что хотя бы одно лицо было найдено
  if len(detections) > 0:
      # предположение, что каждое изображение имеет только ОДНО лицо,
      # что позволяет найти рамку лица с наибольшей вероятностью
      i = np.argmax(detections[0, 0, :, 2])
      confidence = detections[0, 0, i, 2]
      # проведение теста минимальной вероятности (помогает отфильтровать слабые обнаружения)
      if confidence > args["confidence"]:
          # вычислить (x, y)-координаты рамки лица и извлечь точки интереса
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
          face = frame[startY:endY, startX:endX]
          # запись кадра на диск
          p = os.path.sep.join([args["output"],
                                "{}.png".format(saved)])
          cv2.imwrite(p, face)
          saved += 1
          print("[INFO] изображение {} сохранено".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
