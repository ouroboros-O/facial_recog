import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import cv2



class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


z = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}

siamese_model = tf.keras.models.load_model('siamesemodel.h5', custom_objects=z)


def preprocess(file_path):
    
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    
    return img

results = []

def compare(model, detection_threshold, verification_threshold):

    for image in os.listdir(os.path.join('fin_data', 'verifications')):
        input_img = preprocess(os.path.join('fin_data', 'inp', 'inp.jpg'))
        validation_img = preprocess(os.path.join('fin_data', 'verifications', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    
    verification = detection / len(os.listdir(os.path.join('fin_data', 'verifications'))) 
    verified = verification > verification_threshold
    
    return results, verified

    

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('v'):

        cv2.imwrite(os.path.join('fin_data', 'inp', 'inp.jpg'), frame)
        results, verified = compare(siamese_model, 0.9, 0.8)
        print(verified)
        output = [1 if x > 0.6 else 0 for x in results]
        print(output)

    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
