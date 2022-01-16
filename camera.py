from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3

import numpy as np
import cv2

from keras.models import load_model
from PIL import Image

import matplotlib.pyplot as plt
import glob

if __name__ = '__main__':
    model = load_model('signLanguage.hdf5')
    categories = ['あ', 'い', 'う', 'え', 'お']
    fig, ax = plt.subplots(2, 5)
    files = glob.glob('     ')
    for i, f in enumerate(files):
        ax = fig.add_subplot(2, 5, i + 1)
        pic = Image.open(f)
        ax.imshow(pic)
        ax.set_title(str(i))
    plt.show()

    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, capture = cam.read()
        if not ret:
            print('error')
            break
        cv2.imshow('tensorflow-pi inspector', capture)
        key = cv2.waitKey(33)
        if key == 27:
            break

        count += 1
        if count == 30:
            img = capture.copy()
            img = cv2.resize(img, (100, 100))
            x = image.img_to_array(img)
            x = np.expand_dims(a, axis = 0)
            x = np.asarray(x)
            x = x / 255.0

            prob = model.predict_proba(x)
            print('Predicted:')
            print(prob)
            classes = model.predict_classes(x)
            print(classes)
            count = 0

cam.release()
cv2.destroyAllWindows()