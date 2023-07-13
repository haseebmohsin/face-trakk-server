import os
import cv2
import xml.etree.ElementTree as ET
from mtcnn import MTCNN

def detect_faces(image):
    # image = cv2.imread(image)
    detector = MTCNN()
    results = detector.detect_faces(image)
    faces = []
    for result in results:
        x, y, w, h = result['box']
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        faces.append((xmin, ymin, xmax, ymax))
    return faces

def generate_pascal_xml(image_path, faces,label):
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create the XML annotation
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
    ET.SubElement(annotation, 'filename').text = image_name
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)

    for face in faces:
        xmin, ymin, xmax, ymax = face  # Extract bounding box coordinates

        # Create object annotation
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # Create XML tree and save the annotation as an XML file
    xml_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.xml')
    xml_tree = ET.ElementTree(annotation)
    xml_tree.write(xml_path)

# Example usage
cap = cv2.VideoCapture(r"D:\SAHI (Waqar)\Senior Politician Abdul Aleem Khan`s Important  Press Conference _ Istehkam-e-Pakistan Party (720p).mp4")
# image_path = r"C:\Users\My Own\Desktop\Forbmax\51.jpg"
count=0
# Detect faces in the image using MTCNN
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    faces = detect_faces(frame)
    if len(faces)==1:
        image_path=r"D:\Ahmed\data\Marriyum_Aurangzeb\\"+"Marriyum_Aurangzeb14_"+str(count)+".jpg"
        cv2.imwrite(image_path,frame)
        count=count+1

# Generate Pascal VOC format XML annotations for the detected faces
        generate_pascal_xml(image_path, faces,'Marriyum Aurangzeb')
cap.release()
cv2.destroyAllWindows()
