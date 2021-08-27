import cv2
import numpy as np
import imutils
from math import sqrt


class BienSo:
    def __init__(self, config, weight):
        self.config = config
        self.weight = weight

    def get_anh(self):
        return cv2.imread(self.img)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers
 
    def cat_bien_so(self, image):
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        net = cv2.dnn.readNet(self.weight, self.config)
        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        if(len(boxes) == 0):
            return None
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = abs(box[0])
            y = abs(box[1])
            w = abs(box[2])
            h = abs(box[3])
            bien_so = image[round(y):round(y+h)+5, round(x):round(x+w)+5]
        return [x, y, bien_so]

    def tim_so(self, img):
        edged = cv2.Canny(img, 30, 200)
        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        list_so = []
        if len(contours) > 0:
            i = 0
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.2 < aspect_ratio < 0.5 and 0.5 < h/img.shape[0] < 0.8:
                    so = img[y:y+h, x:x+w]
                    so = np.pad(so, 20)
                    so = cv2.GaussianBlur(so, (3, 3), 0)
                    kernel = np.ones((3, 3), np.uint8)
                    dilation = cv2.dilate(so, kernel=kernel)
                    erosion = cv2.erode(dilation, kernel=np.ones(
                        (5, 5), np.uint8), iterations=1)
                    so = cv2.resize(erosion, (28, 28))
                    list_so.append([x, so])

        list_so = sorted(list_so, key=lambda x: x[0])
        return [i[1] for i in list_so]

    def lay_gia_tri(self, img):
        bien_so = self.cat_bien_so(img)
        image = cv2.cvtColor(bien_so[2], cv2.COLOR_BGR2GRAY)

        thre = cv2.threshold(
            image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thre = cv2.bitwise_not(thre)
        thre = imutils.resize(thre, width=400)

        kernel = np.ones((3, 3), np.uint8)

        dilation = cv2.dilate(thre, kernel=kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel=kernel, iterations=1)

        blur = cv2.medianBlur(dilation, 5)

        h, w = blur.shape
        img_tren = blur[:h//2+10, :]
        img_duoi = blur[h//2-10:, :]

        list_tren = self.tim_so(img_tren)
        list_duoi = self.tim_so(img_duoi)

        return list_tren, list_duoi
