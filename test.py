import pickle
import numpy as np
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
#to use sklearn.neighbors we need to install 'sci-kit learn'
from sklearn.neighbors import NearestNeighbors
#to display 5 closet images we will use open CV module
import cv2

features_list = np.array(pickle.load(open('embeddings.pkl','rb')))
#filenames will have the 5 closet image to input image
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
model.trainable = False


model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/58450.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis = 0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

#NearestNeighbor is an algorithm which is used to compare features of input image and images from database.
neighbors = NearestNeighbors(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
neighbors.fit(features_list)

distance, indices = neighbors.kneighbors([normalized_result])

print(indices)

#0th image shown will be input image, we want 5 recommendations of input image
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)