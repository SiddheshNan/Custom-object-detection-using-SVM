import dlib
import cv2
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS

webcam = WebcamVideoStream(src=int(0)).start()
fps = FPS().start()

print('cam started...')

bottle_detector = dlib.simple_object_detector('model/bottle.svm')
phone_detector = dlib.simple_object_detector('model/phone.svm')

while True:
    image = webcam.read()
    image = imutils.resize(image, width=400)

    nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bottles = bottle_detector(nimage)
    phones = phone_detector(nimage)

    for b in bottles:
        # print('bottle detected')
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.putText(image, 'bottle', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    for b in phones:
        # print('phone detected')
        (x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
        cv2.putText(image, 'phone', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)


    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    fps.update()
