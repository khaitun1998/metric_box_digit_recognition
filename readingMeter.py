import numpy as np
import scipy
import cv2
import tensorflow as tf
import keras
import torch
import torchvision
import sklearn
import skimage
import math
import os, logging

from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.contrib.util.constant_value(tf.ones([1]))

class Reader:

    def __init__(self):
        self.name = "Reader"
        self.counting = 0

    # Prepare your models
    def prepare(self):
        self.model_2 = model_builder_task2()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                self.model = model_builder()
        # print("Loaded task 2 model\n")

    # Implement the reading process here
    def process(self, img):
        number_box = crop_number_box(self.model_2, img)
        cv2.imwrite("tmp/" + str(self.counting) + ".jpg", number_box)
        self.counting += 1
        cropped_img = crop_image(number_box)
        if cropped_img == False:
            return 500
        # CNN
        y_pred = []
        for i in range(len(cropped_img)):
            with self.graph.as_default():
                with self.session.as_default():
                    y_pred.append(self.model.predict(cropped_img[i]))
        pred = []
        for i in range(len(y_pred)):
            pred.append(np.argmax(y_pred[i]))

        # All
        pred = np.array(pred)
        result = 0
        for i in pred:
            result *= 10
            result += i
        return result

    # Prepare your models
    def prepare_crop(self):
        # CNN
        self.model = model_builder()
#         print("Loaded task 1 model\n")

    # Implement the reading process here
    def crop_and_process(self, img):
        # All
        cropped_img = crop_image(img)

        # CNN
        y_pred = []
        for i in range(len(cropped_img)):
            y_pred.append(self.model.predict(cropped_img[i]))
        pred = []
        for i in range(len(y_pred)):
            pred.append(np.argmax(y_pred[i]))

        # All
        pred = np.array(pred)
        result = 0
        for i in pred:
            result *= 10
            result += i
        return result


def model_builder():
    model = keras.models.load_model(os.path.abspath("models/metric_digit_recognize/mobilenet_model"))

    return model


def model_builder_task2():
    model_path = os.path.abspath('models/metric_digit_recognize/ssd7_model')

    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    K.clear_session()

    model = keras.models.load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                                'L2Normalization': L2Normalization,
                                                                'DecodeDetections': DecodeDetections,
                                                                'compute_loss': ssd_loss.compute_loss})

    return model


def crop_number_box(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height = 200
    img_width = 200

    orig_images = []
    input_images = []

    orig_images.append(img)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)
    confidence_threshold = 0.1
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    box = y_pred_thresh[0][0]
    xmin = int(round(box[2] * orig_images[0].shape[1] / img_width))
    ymin = int(round(box[3] * orig_images[0].shape[0] / img_height))
    xmax = int(round(box[4] * orig_images[0].shape[1] / img_width))
    ymax = int(round(box[5] * orig_images[0].shape[0] / img_height))

    ret_img = orig_images[0][ymin:ymax, xmin:xmax]

    return ret_img


def crop_image(img):
    try:
        # convert to gray
        img_gray = image2gray(img)

        norm_img = skimage.exposure.rescale_intensity(img_gray)

        # blur image
        blurred_img = cv2.fastNlMeansDenoising(norm_img, None, 8, 7, 21)

        # rotate img
        rotated_image = rotate_img(blurred_img)

        # resize img to 300x100
        resized_img = resize_image(rotated_image)

        # crop image to 5 part
        img_height, img_width = resized_img.shape[:2]
        cropped_img = []
        ret_img = []

        for j in range(0, img_width, img_width // 6):
            cropped_img.append(resized_img[0:img_height, j:j + img_width // 6])

        for k, image in enumerate(cropped_img[:5]):
            # SVM
            # img = cv2.resize(image, (28, 28))
            # img = img.flatten()
            # CNN
            img = cv2.resize(image, (32, 32))

            img = img.reshape((1, 32, 32, 1)).astype('float32')
            img /= 255
            img = np.subtract(img, 0.5)
            img = np.multiply(img, 2.0)
            # img = np.expand_dims(img, axis=2)
            # img = np.expand_dims(img, axis=0)
            ret_img.append(img)
    except:
        return False

    return ret_img


# Fixed
def rotate_img(img):
    img_edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=2500)

    angles = []
    max_dist = 0
    saved_x1, saved_y1, saved_x2, saved_y2 = 0, 0, 0, 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if calculateDistance(x1, y1, x2, y2) > max_dist:
                max_dist = calculateDistance(x1, y1, x2, y2)
                saved_x1, saved_y1, saved_x2, saved_y2 = x1, y1, x2, y2

    angle = math.degrees(math.atan2(saved_y2 - saved_y1, saved_x2 - saved_x1))
    angles.append(angle)

    median_angle = np.median(angles)

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def image2gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray


def resize_image(img):
    img_resize = cv2.resize(img, (300, 100))

    return img_resize


def blur_image(img, level):
    img_blur = cv2.medianBlur(img, level)

    return img_blur


def check_import():
    print("Python 3.6.7")
    print("Numpy = ", np.__version__)
    print("Scipy = ", scipy.__version__)
    print("Opencv = ", cv2.__version__)
    print("Tensorflow = ", tf.__version__)
    print("Keras = ", keras.__version__)
    print("pytorch = ", torch.__version__)
    print("Torch vision = ", torchvision.__version__)
    print("Scikit-learn = ", sklearn.__version__)
    print("Scikit-image = ", skimage.__version__)


if __name__ == "__main__":
    check_import()

"""
Using TensorFlow backend.
Python 3.6.7
Numpy =  1.14.5
Scipy =  1.2.1
Opencv =  4.1.1
Tensorflow =  1.14.0
Keras =  2.3.0
pytorch =  1.0.1.post2
Torch vision =  0.2.2
Scikit-learn =  0.21.3
Scikit-image =  0.14.2
"""
