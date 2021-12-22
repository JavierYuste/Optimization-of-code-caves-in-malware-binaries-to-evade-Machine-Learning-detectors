import os
import tensorflow as tf
import numpy as np
import threading
import logging
from json import encoder
from keras import metrics
from tensorflow.keras.optimizers import SGD
from keras.models import load_model
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d %I:%M:%S', level=logging.DEBUG)

class malconv():
    def __init__(self, NN):
        self.batch_size = 100
        self.input_dim = 257
        self.padding_char = 256
        self.restore_NN(NN)
        self.semaphore = threading.BoundedSemaphore(value=1)

    def restore_NN(self, NN):
        if os.path.exists(NN):
            logging.info("Restoring malconv.h5 from disk...")

            basemodel = load_model(NN)
            self.graph = tf.compat.v1.get_default_graph()
            _, self.maxlen, embedding_size = basemodel.layers[1].output_shape

            logging.info("Restored malconv.h5")

            logging.debug(basemodel.summary())
            self.model = basemodel

            self.model.compile(loss='binary_crossentropy',
                               optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-3),
                               metrics=[metrics.binary_accuracy])
        else:
            logging.error("Could not find malconv.h5")

    def predict(self, binary):
        return self.predict_aux(binary)[0][0]

    def predict_aux(self, binary):
        X = self.another_generator(binary)
        prediction = self.model.predict_generator(X, 1)
        return prediction

    def another_generator(self, binary):
        X = []
        x = self.bytez_to_numpy(binary)
        X.append(x)
        yield np.asarray(X, dtype=np.uint16)

    def bytez_to_numpy(self, binary):
        with open(binary, "rb") as f:
            bytes_ = f.read()
            b = np.ones((self.maxlen,), dtype=np.uint16) * self.padding_char
            bytez = np.frombuffer(bytes_[:self.maxlen], dtype=np.uint8)
            b[:len(bytez)] = bytez
        return b