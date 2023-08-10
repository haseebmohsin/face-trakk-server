#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms
import time
import os


#other lib
import sys
import numpy as np
import base64 
import cv2
import pandas as pd
from datetime import datetime
import json
from moviepy.editor import VideoFileClip, AudioFileClip
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "scripts/yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables to store the detected faces and labels for the previous fully processed frame
prev_frame_faces = []
prev_frame_labels = []
person_data = []

# Get model detect
## Case 1:
# model = attempt_load("yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load("scripts/yolov5_face/yolov5m-face.pt", map_location=device)

# Get model recognition
## Case 1: 
from insightface.insight_face import iresnet100
weight = torch.load("scripts/insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

## Case 2: 
# from insightface.insight_face import iresnet18
# weight = torch.load("insightface/resnet18_backbone.pth", map_location = device)
# model_emb = iresnet18()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

isThread = True
score = 0
name = null

# Resize image
def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 256
    conf_thres = 0.8
    iou_thres = 0.8
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "scripts/static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image, images_names, images_embs):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    return name, score

def time_str(total_seconds):
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return timestamp_str

def time_to_seconds(timestamp_str):
    try:
        hours, minutes, seconds = map(int, timestamp_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except ValueError:
        raise ValueError("Invalid timestamp format. Use hh:mm:ss")
    
def numpy_array_to_base64(image_array, format='.jpg'):
    _, buffer = cv2.imencode(format, image_array)
    base64_image = base64.b64encode(buffer).decode()
    return base64_image

def extract_audio(video_path, audio_output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)
    audio_clip.close()
    video_clip.close()
    
def merge_audio_into_video(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # Set the audio of the video clip to the provided audio clip
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the merged video with audio to the output path
    video_clip.write_videofile(output_path)
    
    audio_clip.close()
    video_clip.close()
    
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")
import glob
def main():
    # global isThread, score, name, prev_frame_faces, prev_frame_labels
    input_path = glob.glob("uploads/videos/*.mp4")
    output_without_audio_path = "scripts/output_without_audio.mp4"
    audio_path = "scripts/audio.mp3"
    output_path = r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk\public\videos\output.mp4"
    # scale_factor = 0.5
    
    # Read features
    images_names, images_embs = read_features()
    
    #create list of timestamps
    label_names = list(set(images_names))
    for n in label_names:
        n = n.replace("_", " ")
        person_entry = {
            'thumbnail': None,
            'name': n,
            'timestamps': [],
            'startTime': [],
            'endTime': [],
            'coverageTime': '00:00:00'
        }
        # Append the dictionary to the list
        person_data.append(person_entry)
    
    # Open video 
    cap = cv2.VideoCapture(input_path[0])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames of the input video
    frame_count = 0
    
    # Save video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(output_without_audio_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, size)
    
    # Add frame interval
    frame_interval = int(output_fps / 3)
    
    # Read until video is completed
    start_total_time = time.time()
    start_time = start_total_time
    #print("cap", cap.isOpened())
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        #print("input video FPS:", output_fps)
        print('Frame#', frame_count, "of", total_frames, "frames")

        # If it's not the frame interval, use the previous frame's data
        if frame_count % frame_interval != 0 and frame_count != 1:
            for box, label in zip(prev_frame_faces, prev_frame_labels):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            video.write(frame)
            new_time = time.time()
            fps = 1 / (new_time - start_time)
            start_time = new_time
            fps_label = "FPS: {:.2f}".format(fps)
            print(fps_label)
            #cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            continue

        # Calculate and display the FPS
        new_time = time.time()
        fps = 1 / (new_time - start_time)
        start_time = new_time
        fps_label = "FPS: {:.2f}".format(fps)
        print(fps_label)
        #cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Get faces
        bboxs, landmarks = get_face(frame)
        # h, w, c = frame.shape
        
        # tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        
        # Get boxs
        prev_frame_faces = []
        prev_frame_labels = []
        # Get the current position of the video capture
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_seconds = int(position_ms / 1000)
        frame_timestamp= time_str(timestamp_seconds)
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            
            # Get recognized name
            face_image = frame[y1:y2, x1:x2]
            name, score = recognition(face_image, images_names, images_embs)
            #print("Detected: ", name, "with score: ", score)
            # # Get recognized name
            # if isThread == True:
            #     isThread = False
                
            #     # Recognition
            #     face_image = frame[y1:y2, x1:x2]
            #     thread = Thread(target=recognition, args=(face_image, images_names, images_embs))
            #     thread.start()

            # # Landmarks
            # for x in range(5):
            #     point_x = int(landmarks[i][2 * x])
            #     point_y = int(landmarks[i][2 * x + 1])
            #     cv2.circle(frame, (point_x, point_y), tl+1, clors[x], -1)

            if name == null:
                continue
            else:
                if score < 0.35:
                    label = "Unknown"
                    prev_frame_labels.append(label)
                    prev_frame_faces.append(bboxs[i])
                    #print("Detected: ", caption, "with score: ", score)
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                    cv2.putText(frame, label, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
                else:
                    label = name.replace("_", " ")
                    for p in person_data:
                        if label == p['name']:
                            p['timestamps'].append(frame_timestamp)
                            if p['thumbnail'] == None:
                                p['thumbnail'] = numpy_array_to_base64(face_image)
                    caption = f"{label}:{score:.2f}"
                    prev_frame_labels.append(label)
                    prev_frame_faces.append(bboxs[i])
                    #print("Detected: ", caption, "with score: ", score)
                    t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
                    cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)       
        
        # Count fps 
        video.write(frame)
        #cv2.imshow("Face Recognition", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
    print("Video without audio saved at: ", output_without_audio_path)
    
    # Extract audio from input video
    extract_audio(input_path[0], audio_path)
    
    # Merge audio into output video
    merge_audio_into_video(output_without_audio_path, audio_path, output_path)
    
    # remaining data
    for p in person_data:
        coverage_time = 0
        if len(p['timestamps']) >= 10:
            p['startTime'].append(p['timestamps'][0])
            for t in range(0,len(p['timestamps'])):
                ts = time_to_seconds(p['timestamps'][t])
                prev_ts = time_to_seconds(p['timestamps'][t-1])
                if ts - prev_ts >= 2:
                    p['startTime'].append(p['timestamps'][t])
                    p['endTime'].append(p['timestamps'][t-1])
            if len(p['startTime']) != len(p['endTime']):
                if len(p['startTime']) > len(p['endTime']):
                    p['endTime'].append(p['timestamps'][-1])
                else:
                    p['endTime'][:-1]
            #print('start',len(p['startTime']))
            #print('end',len(p['endTime']))
            for tt in range(0, len(p['startTime'])):
                tts = (time_to_seconds(p['endTime'][tt]) - time_to_seconds(p['startTime'][tt]))
                coverage_time = coverage_time + tts
            p['coverageTime'] = time_str(coverage_time)
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(person_data)
    condition = df['timestamps'].apply(len) >= 10
    filtered_df = df[condition]
    print("data:", filtered_df)
    
    # DataFrame to .json file
    file_path = "scripts/data.json"
    json_data = filtered_df.to_json(orient='records')
    
    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    print("Data saved to:", file_path)
  
    # # Define the output CSV file path
    # output_csv_path = "output_videos/data.csv"

    # # Save the DataFrame to a CSV file
    # filtered_df.to_csv(output_csv_path, index=False)
    # print(f"DataFrame saved to '{output_csv_path}'.")
    
    # Delete temporary files
    delete_file(output_without_audio_path)
    delete_file(audio_path)
    
    end_total_time = time.time()
    total_time = end_total_time - start_total_time
    print("Total processing time = ", total_time)
    return json_data


if __name__=="__main__":
    main()