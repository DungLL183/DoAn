import os
import cv2
import numpy as np
# define the paths to the positive and negative training images
# pos_path = 'positive_faces/'
# neg_path = 'negative_faces/'
pos_path = 'data'
neg_path = 'negative_faces'

# create a list of positive and negative images
pos_images = [pos_path + i for i in os.listdir(pos_path)]
neg_images = [neg_path + i for i in os.listdir(neg_path)]

# create a list of labels for the positive and negative images
pos_labels = [1] * len(pos_images)
neg_labels = [0] * len(neg_images)

# print(pos_labels)
# print(neg_labels)
# combine the positive and negative images and labels into a single dataset
images = pos_images + neg_images
labels = pos_labels + neg_labels

# define the haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# define the function to extract features from the images
def extract_features(image):
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
  # if no faces are detected, return none
  if len(faces) == 0:
    return None
  
  # extract the face region from the image and resize it to a common size
  x,y,h,w = faces[0]
  face = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
  
  # return the face region as a feature vector
  return face.flatten()
 
# extract features from the images and store them in a feature matrix 
features = []
for img_ in pos_images:
    feature = extract_features(img_)
    if feature is not None:
        features.append(feature)

# convert the features and labels to numpy arrays
features = np.array(features)

# define the svm classifier and train it on the features and labels
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# save the train classifier to a file
svm.save('face_classifier.xml')

# ====================================================================== LBPH ===================================================================================

# To create a facial classifier train file using local binary pattern histogram (LBPH) algorithm, we need to follow these steps:
# 1. import the required libraries : opencv, numpy, and os
# 2. define the image directories for positive and negative training data
# 3. initialize two empty lists for storing the positive and negatvie training data
# 4. define a function to read the images from the directories and convert them to grayscale
# 5. apply the lbph algorithm on the grayscale images and store the result in the respective list
# 6. concatenate the positive and negative training data lists
# 7. convert the lists to a numpy array and save it to a file

# import cv2
# import numpy as np
# import os

# # define the image directories to positive and negative training data
# pos_dir = 'positive_faces/'
# neg_dir = 'negative_faces/'

# # initialize empty lists for storing positive and negative training data
# pos_data = []
# neg_data = []

# # function to read images and convert them to grayscale 
# def read_images(dir_name):
#     for file_name in os.listdir(dir_name):
#         img = cv2.imread(dir_name + file_name)
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         yield gray_img

# # apply LBPH algorithm on grayscale images and store thr result in the perspective list
# for gray_img in read_images(pos_dir):
#     lbph = cv2.face.LBPHFaceRecognizer_create()
#     lbph.train(np.array([gray_img]), np.array([1]))
#     pos_data.append(lbph.getHistogram())
    
# for gray_img in read_images(neg_dir):
#     lbph = cv2.face.LBPHFaceRecognizer_create()
#     lbph.train(np.array([gray_img]), np.array([1]))
#     pos_data.append(lbph.getHistogram())

# # concatenate the positive and negative training data lists
# train_data = np.concatenate(pos_data, neg_data)

# # save the training data to a file
# np.save('train_data.npy', train_data)

# # in this code, we first define the directories for positive and negative training data. we then use "read_images" funtion to read the images from directories 
# # and convert tem to grayscale. we apply the lbph algorithm on the grayscale images using the "cv2.face.LBPHFaceRecognizer" method and store the result in the respective list
# # finally we concatenate the positive and negative training data lists and save the result to a file using the "np.save" method


# # ==================================================================================== ANOTHER =========================================================================
# import cv2
# import numpy as np

# # Prepare the data
# positive_images = []
# negative_images = []
# positive_labels = []
# negative_labels = []

# # Load and label the positive images
# # ... 
# # Load and label the negative images
# # ... 

# # Split the data into training and testing sets
# X_train = np.concatenate((positive_images[:80], negative_images[:80]), axis=0)
# X_test = np.concatenate((positive_images[80:], negative_images[80:]), axis=0)
# y_train = np.concatenate((positive_labels[:80], negative_labels[:80]), axis=0)
# y_test = np.concatenate((positive_labels[80:], negative_labels[80:]), axis=0)

# # Train the classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# lbph_face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Convert the images to grayscale
# gray_train = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train]
# gray_test = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_test]

# # Train the Haar Cascade classifier
# face_samples = []
# for img in gray_train:
#     faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         face_roi = img[y:y+h, x:x+w]
#         face_samples.append(face_roi)
#         face_labels = np.array(y_train)
# face_cascade.train(face_samples, face_labels)

# # Train the LBPH face recognizer
# lbph_face_recognizer.train(gray_train, np.array(y_train))

# # Save the trained classifiers
# face_cascade.save('face_cascade.xml')
# lbph_face_recognizer.save('lbph_face_recognizer.xml')

# # Test the classifier
# face_cascade.load('face_cascade.xml')
# lbph_face_recognizer.load('lbph_face_recognizer.xml')

# # Evaluate the performance of the classifier
# correct = 0
# total = 0
# for i, img in enumerate(gray_test):
#     faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         face_roi = img[y:y+h, x:x+w]
#         label, confidence = lbph_face_recognizer.predict(face_roi)
#         if label == y_test[i]:
#             correct += 1
#         total += 1
#         accuracy = correct / total
# print("Accuracy:", accuracy)
