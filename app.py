import tensorflow
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
#pickle - Serialization is the process of converting a Python object into
# a byte stream, and deserialization is the reverse process of reconstructing
# a Python object from a byte stream.
#The primary use of pickle is to store Python objects in a format that can
# be saved to a file or transmitted over a network and later reconstructed.


model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False


model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    #loads an image from a specified path and resizes it to target size
    # of 224 X 224
    img = image.load_img(img_path, target_size=(224,224))
    #to convert image into numpy array
    img_array = image.img_to_array(img)
    #it adds the dimensions in order to create a batch of one image
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    #preprocess_input function is used to make image suitable for
    # ResNET50 model
    preprocessed_img = preprocess_input(expanded_img_array)
    #it flattens the result into 1D array
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

#create a list of file paths by iterating through the files
# in the 'images' directory and appending the file paths to the
# filenames list.
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))


#These lines create an empty list called feature_list and then loop
# through each file path in filenames. For each file, it calls the
# extract_features function to extract features using the
# ResNet50 model and appends the result to the feature_list list.
feature_list = []

#tqdm is used to display the progress bar during the loop to track the progress of the image
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))