from imutils import paths
import imutils
import face_recognition
import cv2
import os
import pickle
import time
from pathlib import Path
import dlib

# print(dlib.DLIB_USE_CUDA)
dlib.DLIB_USE_CUDA=True
Path("models").mkdir(parents=True, exist_ok=True)
print('[INFO] creating facial embeddings...')

ti = time.time()
knownEncodings, knownNames = [], []
# print(paths.list_images(os.getcwd() + '//Dataset_cropped'))
# print(os.getcwd())
imagePaths = list(paths.list_images(os.getcwd() + '//Dataset'))  # dataset here
# imagePaths=r"D:\SAHI (Waqar)\Face-Recognition-main\Dataset_cropped")

for (i, imagePath) in enumerate(imagePaths):
	# print(i)
    print('{}/{}'.format(i+1, len(imagePaths)), end=', ')
    print(i)
    # print("image done ")
    image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(
        rgb,  model='cnn')  # detection_method here
    for encoding in face_recognition.face_encodings(rgb, boxes, model='large'):
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {'encodings': knownEncodings, 'names': knownNames}
f = open(os.getcwd() + '\\models\\model.pickle', 'wb')
f.write(pickle.dumps(data))
f.close()
print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))
