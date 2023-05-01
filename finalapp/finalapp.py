import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
import uuid

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger



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


def compare(model, detection_threshold, verification_threshold):
    
    results = []

    for image in os.listdir(os.path.join('ai_app_data', 'verifications')):
        input_img = preprocess(os.path.join('ai_app_data', 'inp', 'inp.jpg'))
        validation_img = preprocess(os.path.join('ai_app_data', 'verifications', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    
    verification = detection / len(os.listdir(os.path.join('ai_app_data', 'verifications'))) 
    print(verification)

    verified = verification > verification_threshold
    
    return results, verified

 

detection_threshold = 0.85
correct_ratio = 0.70
global time
time = 0


class AiApp(App):
    def build(self):
       
        # Main layout components 
        self.logo = Button(text="TEACHING THE SAND TO THINK", size_hint=(1,.2), background_color=[255, 0, 0,255], font_size=45)
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Click here to Verify", on_press=self.verify, size_hint=(1,.1), background_color=[255, 0, 71,255], font_size=25)
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))
        self.button2 = Button(text="Click here to authenticate new user", on_press=self.neww, size_hint=(1,.1))
        self.verification_label.font_size = 25

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.logo)

        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.web_cam)


        self.instruc = Label(text="INSTRUCTIONS FOR NEW: LOOK AT THE CAMERA AND MOVE YOUR HEAD AROUND UNTIL TOLD TO STOP", size_hint=(1,.1))
        layout.add_widget(self.instruc)

        layout.add_widget(self.button2)

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    def update(self, *args):

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def takepicture(self, *args):
            ret, frame = self.capture.read()

            frame = frame[120:120+250, 200:200+250, :]
            imgname = os.path.join(os.path.join('ai_app_data', 'verifications'), '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

    def ontimeout(self, *args):
            self.instruc.text = 'Done! New user acess granted.'
            self.verification_label.color = (0, 1, 0, 1)
            self.event.cancel()


    def neww(self, *args):
        nuke_path = os.path.join('ai_app_data', 'verifications')

        for filename in os.listdir(nuke_path):
            file_path = os.path.join(nuke_path, filename)
            os.remove(file_path)
        time = 0
        self.instruc.text = 'Please wait...'

        self.event = Clock.schedule_interval(self.takepicture, 1/15.0)
        Clock.schedule_once(self.ontimeout, 8.0)


    def verify(self, *args):
        SAVE_PATH = os.path.join('ai_app_data', 'inp', 'inp.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results, verified = compare(siamese_model, detection_threshold, correct_ratio)
        output = [1 if x > 0.6 else 0 for x in results]
        print(output)
        print(verified)
        if verified:
            self.verification_label.text = 'Verified'
            self.verification_label.color = (0, 1, 0, 1)
        else:
            self.verification_label.text = 'Not Verified'
            self.verification_label.color = (1, 0, 0, 1) 

        print(verified)


        return
if __name__ == '__main__':
    ap = AiApp()
    ap.run()