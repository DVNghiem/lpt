from YOLO import BienSo
import cv2
import numpy as np
from imutils import resize
from load_model import Model_Mnist, Model_Written

mnist = Model_Mnist()
written = Model_Written()

mnist.load_model('model_mnist/model.ckpt.meta', 'model_mnist')
written.load_model('model_hand_written/model.ckpt.meta', 'model_hand_written')


b = BienSo('yolov4-tiny.cfg', 'yolov4-tiny_3000.weights')
def image(img_path):
    img = cv2.imread(img_path)
    bien_so = b.cat_bien_so(img)
    list_tren, list_duoi = b.lay_gia_tri(img)
    result = ''
    for i in list_tren[:2]:
        i = i.reshape(1, 784)
        pred = mnist.predict(i)
        result += str(pred)
    result += '-'

    chu = list_tren[-2]
    chu = chu.reshape(1, 784)
    pred = written.predict(chu)
    result += chr(pred+65)

    so_cuoi = list_tren[-1]
    so_cuoi = so_cuoi.reshape(1, 784)
    pred = mnist.predict(so_cuoi)
    result += str(pred)
    result += '/'
    for i in list_duoi:
        i = i.reshape(1, 784)
        pred = mnist.predict(i)
        result += str(pred)

    cv2.putText(img, result, (int(bien_so[0]), int(
        bien_so[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Bien so xe', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image('test1.jpg')
    
