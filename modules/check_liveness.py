# импорт необходимых пакетов
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import dlib
import time
# для рассчета расстояния между точками интереса
from scipy.spatial import distance as dist
# получение номеров меток глаз
from imutils import face_utils

protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5
model = load_model("./models/liveness.model")
le = pickle.loads(open("./models/le.pickle", "rb").read())
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def check_texture(face_image):
  face_image = cv2.imread(face_image)

  print("[INFO] получение кадра...")
  frame = face_image
  frame = imutils.resize(frame, width=600)

  # получение размеров кадра и преобразование
  print("[INFO] преобразование кадра...")
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

  # передать blob через модель для получения обнаружений и прогнозов
  print("[INFO] пропуск blob через сеть...")
  net.setInput(blob)
  detections = net.forward()

  # цикл по обнаружениям
  print("[INFO] цикл по обнаружениям...")
  for i in range(0, detections.shape[2]):
    # извлечь достоверность (т. е. вероятность), связанную с прогнозом
    confidence_detected = detections[0, 0, i, 2]

    # отфильтровывать слабые обнаружения
    if confidence_detected > confidence:
      # вычислить (x, y)-координаты рамки лица и извлечь точки интереса
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      # проверка, что обнаруженная рамка выходит за пределы размеров кадра
      startX = max(0, startX)
      startY = max(0, startY)
      endX = min(w, endX)
      endY = min(h, endY)

      # извлечение точек интереса, предварительная обработка по аналогии с обучающими данными.
      face = frame[startY:endY, startX:endX]
      face = cv2.resize(face, (32, 32))
      face = face.astype("float") / 255.0
      face = img_to_array(face)
      face = np.expand_dims(face, axis=0)

      # пропустить точки интереса лица через обученную модель детектора живости, для определения,
      # является ли лицо «настоящим» или «фальшивым».
      preds = model.predict(face)[0]
      j = np.argmax(preds)
      decision = le.classes_[j]
      if decision == "fake":
        print("texture ", decision)
        return False
      else:
        print("texture ", decision)
        return True


# check by frames
# NOT VALID
def check_blink():
  # Initializing the face and eye cascade classifiers from xml files
  face_cascade = cv2.CascadeClassifier('./face_detector/haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('./face_detector/haarcascade_eye_tree_eyeglasses.xml')

  # Variable store execution state
  first_read = True
  blink_detected = False

  # Starting the video capture
  cap = cv2.VideoCapture(0)
  ret, img = cap.read()

  while ret:
    ret, img = cap.read()
    # Converting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    if len(faces) > 0:
      for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # roi_face is face which is input to eye classifier
        roi_face = gray[y:y + h, x:x + w]
        roi_face_clr = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_face, 1.2, 4)

        # Examining the length of eyes object for eyes
        if len(eyes) >= 2:
          # Check if program is running for detection
          if first_read:
            cv2.putText(img,
                        "Press E to begin",
                        (70, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 0), 2)
          else:
            cv2.putText(img,
                        "Eyes open, blink!", (70, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 2)
        else:
          if first_read:
            # To ensure if the eyes are present before starting
            cv2.putText(img,
                        "No eyes detected", (70, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 255), 2)
          else:
            # This will print on console and restart the algorithm
            print("Blink detected--------------")
            original = img
            cv2.imwrite("./resources/face_last_login.png", original)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            return True

    else:
      cv2.putText(img,
                  "No face detected", (100, 100),
                  cv2.FONT_HERSHEY_PLAIN, 3,
                  (0, 255, 0), 2)

    # Controlling the algorithm with keys
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if (a == ord('q')):
      break
    elif (a == ord('e') and first_read):
      # This will start the detection
      first_read = False

  cap.release()
  cv2.destroyAllWindows()


# проверка моргания при помощи EAR (Eye Aspect Ratio) Соотношение сторон глаза
def blinking():
  # определение функции для расчета EAR
  def calculate_EAR(eye):
    # рассчитать вертикальное расстояние
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # рассчитать горизонтальное расстояние
    x1 = dist.euclidean(eye[0], eye[3])

    # рассчитать EAR
    EAR = (y1 + y2) / x1
    return EAR

  # переменные
  blink_thresh = 0.45 # пороговое значение мигания (зависит от частоты кадров камеры)
  succ_frame = 2 # через сколько кадров идет каждая проверка
  count_frame = 0 # счетчик кадров

  # ориентиры (точки интереса) глаз
  (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
  (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

  # инициализация модели для точек интереса и обнаружения лица
  detector = dlib.get_frontal_face_detector()
  landmark_predict = dlib.shape_predictor(
    './face_detector/shape_predictor_68_face_landmarks.dat')
  # запуск видеопотока с камеры
  cam = cv2.VideoCapture(0)
  # запуск цикла обнаружения
  while cam.isOpened():
    # если видео закончилось само, сбросить его до начала
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(
            cv2.CAP_PROP_FRAME_COUNT):
      cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # запись с камеры в ином случае
    else:
      ret, frame = cam.read()
      if ret:
        assert not isinstance(frame, type(None)), 'Нет возможности записать кадр'
      cv2.putText(frame, 'Just Blink', (30, 30),
                  cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
      frame = imutils.resize(frame, width=640)
      # преобразование кадра в шкалу серого для передачи на детектор
      img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # обнаружение лиц
      faces = detector(img_gray)
      for face in faces:

        # обнаружение точек интереса
        shape = landmark_predict(img_gray, face)

        # преобразование класса формы непосредственно в список координат (x, y)
        shape = face_utils.shape_to_np(shape)

        # парсинг списка точек интереса для точек левого и правого глаза
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]

        # рассчет EAR
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)

        # среднее значение EAR для левого и правого глаза
        avg = (left_EAR + right_EAR) / 2
        if avg < blink_thresh:
          count_frame += 1  # увеличение количества кадров
        else:
          if count_frame >= succ_frame:
            cv2.putText(frame, 'Blink Detected', (30, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            time.sleep(0.3)
            ret, original = cam.read()
            cv2.imwrite("./resources/face_last_login.png", original)
            cam.release()
            cv2.destroyAllWindows()
            return True
          else:
            count_frame = 0

      cv2.imshow("Just Blink", frame)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

  cam.release()
  cv2.destroyAllWindows()
