from imutils import paths
import face_recognition, cv2, os, pickle, time
from collections import Counter

ti = time.time()
print('Creating Facial Embeddings...')
try:
    trained_data = pickle.loads(open(os.getcwd() + '/encodings.pickle', 'rb').read())    #encodings here
    print("trained_data", trained_data)
except FileNotFoundError:
    knownEncodings, knownNames = [], []
    imagePaths = list(paths.list_images(os.getcwd() + '/dataset'))    #trained_dataset here
    print("imagePaths",imagePaths)
    for (i, imagePath) in enumerate(imagePaths):
        print('{}/{}'.format(i+1, len(imagePaths)), end=', ')
        image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,  model='cnn')    #detection_method here
        for encoding in face_recognition.face_encodings(rgb, boxes):
            knownEncodings.append(encoding)
            knownNames.append(name)
    trained_data = {'encodings': knownEncodings, 'names': knownNames}
    print("Data is here ------",trained_data)
    f = open(os.getcwd() + '/encodings.pickle', 'wb')
    f.write(pickle.dumps(trained_data))
    f.close()

print('Done! \nRecognising Faces in Image...')
imagePaths = list(paths.list_images(os.getcwd() + '/image_test'))    #test image here
for (_, imagePath) in enumerate(imagePaths):
    if '_output' not in imagePath:
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='cnn')    #detection_method here
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for encoding in encodings:
            votes = face_recognition.compare_faces(trained_data['encodings'], encoding)
            if True in votes:
                names.append(Counter([name for name, vote in list(zip(trained_data['names'], votes)) if vote == True]).most_common()[0][0])
            else:
                names.append('Unknown')
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(imagePath.rsplit('.', 1)[0] + '_output.jpg', image)

print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))
