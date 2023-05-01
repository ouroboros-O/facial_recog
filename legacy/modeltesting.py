import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer




gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)



POS_PATH = os.path.join('data', 'pos')
NEG_PATH = os.path.join('data', 'neg')
ANC_PATH = os.path.join('data', 'anc')

anc = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(500)
pos = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(500)
neg = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(500)

def preprocess(file_path):
    
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    
    return img
positives = tf.data.Dataset.zip((anc, pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(anc)))))
negatives = tf.data.Dataset.zip((anc, neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anc)))))                                 
data = positives.concatenate(negatives)


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


samples = data.as_numpy_iterator()
exampple = samples.next()



data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

    




class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)



z = {'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy}
siamese_model = tf.keras.models.load_model('siamesemodel.h5', custom_objects=z)



test_input, test_val, y_true = test_data.as_numpy_iterator().next()
output_array = siamese_model.predict([test_input, test_val])
print(siamese_model.predict([test_input, test_val]))
output = [1 if prediction > 0.5 else 0 for prediction in output_array]
print(output)
siamese_model.summary()

i = 0

plt.figure(figsize=(13,13))

plt.subplot(1,2,1)
plt.imshow(test_input[i])

plt.subplot(1,2,2)
plt.imshow(test_val[i])
if(output[i]==1):
    plt.text(100, 100, "YES", fontdict=None, c="green", fontsize="xx-large")
else:
    plt.text(100, 100, "NO", fontdict=None, c="red", fontsize="xx-large")
plt.text(-30, 130, "IS THAT SUHAIBbBB??", fontdict=None, c="black", fontsize="xx-large")

plt.show()

while i<16:
    i = i + 1
    plt.figure(figsize=(10,10))

    plt.subplot(1,2,1)
    plt.imshow(test_input[i])

    plt.subplot(1,2,2)
    plt.imshow(test_val[i])
    if(output[i]==1):
        plt.text(100, 100, "YES", fontdict=None, c="green", fontsize="xx-large")
    else:
        plt.text(100, 100, "NOT", fontdict=None, c="red", fontsize="xx-large")
    plt.text(-30, 130, "IS THAT SUHAIBbBB??", fontdict=None, c="black", fontsize="xx-large")



    plt.show()





