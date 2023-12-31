# from imutils import paths
# from imutils.video import VideoStream
import imutils
import face_recognition
import cv2
import os
import pickle
# import time
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
import shutil
import re
import argparse
from clustering import clustering_design
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
    i=0
    for face in faces:
	    # top, right, bottom, left = face_box

	    # # Print or use the coordinates as needed
	    # print("xmin:", left)
	    # print("xmax:", right)
	    # print("ymin:", top)
	    # print("ymax:", bottom)


        ymin, xmax, ymax, xmin = face  # Extract bounding box coordinates

        # Create object annotation

        # if "Unknown" in label[i]:
        #     result=label[i]
        #     # print(result)
        # else:
        #     result = re.sub(r'\d+$', '', label[i])
        	# print(result)
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label[i]
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        i=i+1
    # Create XML tree and save the annotation as an XML file
    xml_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.xml')
    xml_tree = ET.ElementTree(annotation)
    xml_tree.write(xml_path)
if os.path.exists("scripts//faces"):
	shutil.rmtree("scripts//faces")
	shutil.rmtree("scripts//clusters")
	shutil.rmtree("scripts//dataset_generate")
Path("scripts//faces").mkdir(parents=True, exist_ok=True)
Path("scripts//dataset_generate").mkdir(parents=True, exist_ok=True)
Path("scripts//clusters").mkdir(parents=True, exist_ok=True)
# ti = time.time()
# print('[INFO] creating facial embeddings...')

data = pickle.loads(
    open(r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\models\model.pickle", 'rb').read())  # encodings here
# print(data)

# print('Done! \n[INFO] recognising faces in webcam...')
# vs = VideoStream(src=0).start()  # access webcam
# time.sleep(2.0)  # warm up webcam
# writer = None
def data_generation_with_xml(video_path):
	cap = cv2.VideoCapture(video_path)
	video_length=cap.get(cv2.CAP_PROP_FRAME_COUNT)
	# print(video_length)
	# Check if camera opened successfully
	if (cap.isOpened()== False):
		print("Error opening video file")
		
	# Read until video is completed

	read_frame_set=0
	counter=0

	# face_and_name={}
	Unknown_person=0
	known_person=0
	while(cap.isOpened()):
		if read_frame_set>=video_length:
			break

	# while True:
		cap.set(cv2.CAP_PROP_POS_FRAMES, read_frame_set) # optional
		dt=datetime.now()
		# print(str(dt))
		# Unknown_person=0
		ret, frame = cap.read()
		# print(frame)
		# print(frame.shape())
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=450)
		# print(rgb.shape)
		r = frame.shape[1] / float(rgb.shape[1])
		boxes = face_recognition.face_locations(
			rgb, model='hog')  # detection_method here
		# print(boxes)
		encodings = face_recognition.face_encodings(rgb, boxes, model='large')
		names = []
		bounding_box_xml=[]
		for encoding in encodings:
			votes = face_recognition.compare_faces(
				data['encodings'], encoding, tolerance=0.1)
			if True in votes:
				names.append(Counter([name for name, vote in list(
					zip(data['names'], votes)) if vote == True]).most_common()[0][0]+str(known_person))
				known_person=known_person+1
			else:
				names.append('Unknown'+str(Unknown_person))
				Unknown_person=Unknown_person+1
		# counter_temp=0
		for ((top, right, bottom, left), name) in zip(boxes, names):
			top, right, bottom, left = int(
				top * r), int(right * r), int(bottom * r), int(left * r)
			padding = 20
			top = max(0, top - padding)
			right = min(frame.shape[1], right + padding)
			bottom = min(frame.shape[0], bottom + padding)
			left = max(0, left - padding)
			# cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
			# y = top - 15 if top - 15 > 15 else top + 15
			# cv2.putText(frame, name, (left, y),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

			# cropped_image = frame[Y:Y+H, X:X+W]
			# # # print(top, right, bottom, left)
			bounding_box_xml.append([top, right, bottom, left])
			# print(boxes[counter_temp],'-->',top, right, bottom, left)
			# counter_temp=counter_temp+1
			cropped_image=frame[top:bottom, left:right]
			cv2.imwrite("scripts//faces//"+name+'.jpg',cropped_image)
	        # break
	    # if writer is None:
	    #     writer = cv2.VideoWriter(os.getcwd() + '\\webcam_test\\output.avi',
	    # 
	                                 # cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]), True)
		if len(names)>0:
			# file_name_save="Waqar.jpg"
			file_name_save="scripts//dataset_generate//"+str(dt.year)+"_"+str(dt.month)+"_"+str(dt.day)+"_"+str(dt.hour)+"_"+str(dt.minute)+"_"+str(dt.second)+"_"+str(dt.microsecond)+".jpg"
			cv2.imwrite(file_name_save,frame)
			generate_pascal_xml(image_path=file_name_save, faces=bounding_box_xml,label=names)
			

		    # counter=counter+1
	    # cv2.imshow('Webcam', frame)
	    # if cv2.waitKey(1) & 0xFF == ord('q'):
	    #     break
	    # time.sleep(30)
		read_frame_set=read_frame_set+50
		# import glob
		# removing_files = glob.glob('faces*.jpg')
		# for i in removing_files:
		#     os.remove(i)
	cv2.destroyAllWindows()
	clustering_design()
	# print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Data Generation with XML')
    
    # Add argument for video path
    parser.add_argument('video_path',type=str, help='Path to the video file')

    # Add argument for frame_number
    # parser.add_argument('frame_number',type=int,default=0,help='frame_number')

    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the data_generation_with_xml function with the provided video path
    data_generation_with_xml(args.video_path)
	
