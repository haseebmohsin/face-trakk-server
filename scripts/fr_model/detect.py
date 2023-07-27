"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import os 
import cv2
import torch
import torch.backends.cudnn as cudnn
import shutil
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
def get_video_time(cap):
    # Get the current position of the video capture in milliseconds
    position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Convert the time in milliseconds to hours, minutes, and seconds
    total_seconds = int(position_ms / 1000)
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60

    return hours, minutes, seconds
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

if os.path.exists(r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\fr_model//thumbnails"):
	shutil.rmtree(r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\fr_model//thumbnails")
	# shutil.rmtree("scripts//clusters")
	# shutil.rmtree("scripts//dataset_generate")
Path(r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\fr_model//thumbnails").mkdir(parents=True, exist_ok=True)
# Path("scripts//dataset_generate").mkdir(parents=True, exist_ok=True)
# Path("scripts//clusters").mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir="scripts/fr_model"

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    count_object=0
    detection_dict = {}
    timestamp_dict={}
    
    for path, img, im0s, vid_cap in dataset:
        hours, minutes, seconds = get_video_time(vid_cap)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # bottom_left_corner = (10, 30)
        # font_scale = 1
        # font_color = (255, 255, 255)  # White color (BGR format)
        # thickness = 2
        video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk\public\videos/output.mp4" # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            if len(det):
                detections = [{'label': names[int(cls)], 'timestamp': time.time()} for *xyxy, conf, cls in reversed(det)]
                for detection in detections:
                    label = detection['label']
                    timestamp = detection['timestamp']
                    # print("video_time,,,,,",video_time)
                    if label in detection_dict:
                        detection_dict[label].append(timestamp)
                        if video_time in timestamp_dict[label]:
                        	pass
                        else:
	                        timestamp_dict[label].append(video_time)
                        # timestamp_dict[label].add(video_time)
                        # print(timestamp_dict)
                    else:
                        detection_dict[label] = [timestamp]
                        timestamp_dict[label]=[video_time]
                        # timestamp_dict[label]={video_time}
                        # print(timestamp_dict)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # count_object=count_object+1
                        # label=''
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            xyxy = torch.tensor(xyxy).view(-1, 4)
                            b = xyxy2xywh(xyxy)  # boxes
                            gain=1.02
                            pad=10
                            b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                            xyxy = xywh2xyxy(b).long()
                            clip_coords(xyxy, imc.shape)
                            BGR=True
                            crop = imc[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
                            # print(save_dir+ "//" + names[c] + '.jpg')
                            cv2.imwrite(r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\fr_model//thumbnails"+ "//" + names[c] + '.jpg',crop)
                            # save_one_box(xyxy, imc, file=save_dir+ "//" + names[c] + '.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            '''
            label video create 
            shah.mp4
            label found then video usy mn jana chaye
                video write
                counter=0
                "shhbas"=0
                "maryam"=+1 
            '''
            # Stream results
            if view_img:
                
                im0=cv2.resize(im0,(700,350))
                # cv2.putText(im0, video_time, bottom_left_corner, font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.putText(
                        im0, #numpy array on which text is written
                        str(count_object), #text
                        (100,200), #position at which writing has to start
                        cv2.FONT_HERSHEY_SIMPLEX, #font family
                        5, #font size
                        (209, 80, 0, 255), #font color
                        5)#font stroke
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # pass
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    # print(f'Done. ({time.time() - t0:.3f}s)')
# After the loop that processes the predictions
    # print("Detection Dictionary:")
    # print(detection_dict)

    # Calculate coverage time for each label (if needed)
    coverage_time = {}
    for label, timestamps in detection_dict.items():
        coverage_time[label] = round((max(timestamps) - min(timestamps))/60,2)

    # print("Coverage Time:")
    # print(coverage_time)
    # print("time stamp details")
    # for key, value in timestamp_dict.items():
    # 	timestamp_dict[key]=set(value)
    # print(timestamp_dict)
    return timestamp_dict,coverage_time

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\scripts\fr_model//ptv_runs//weights//best.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"C:\Users\haseeb\Desktop\Forbmax Applications\facetrakk-app\facetrakk-server\uploads\videos", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', type=str,default=True, help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    import json 
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    timestamp_dict, coverage_time = run(**vars(opt))

    # Combine both dictionaries into a single dictionary
    result_dict = {
        "timestamp_dict": timestamp_dict,
        "coverage_time": coverage_time
    }
    # print(result_dict)
    # Convert the new dictionary to JSON
    # json_data = json.dumps(result_dict)

    # Print the JSON string or use it as needed
    # print(json_data)
    file_path = "scripts/fr_model/data.json"
    # Create lists to store the data for each person
    names = []
    start_times = []
    end_times = []
    coverage_times = []
    import pandas as pd 
    import base64
    from PIL import Image
    # Extract data from the dictionary and populate the lists
    # Create lists to store the data for each person
    names = []
    start_times = []
    end_times = []
    coverage_times = []
    base64_images = []
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                base64_string = base64.b64encode(image_bytes).decode("utf-8")
                return base64_string
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    # def image_to_base64(image_path):
    #     try:
    #         # Open the image using Pillow
    #         with Image.open(image_path) as img:
    #             # Convert the image to bytes
    #             img_bytes = img.tobytes()

    #             # Encode the image bytes to Base64
    #             base64_encoded = base64.b64encode(img_bytes).decode('utf-8')

    #             return base64_encoded
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return None

    # Extract data from the dictionary and populate the lists
    for name, timestamps in result_dict['timestamp_dict'].items():
        names.append(name)
        start_times.append(timestamps[0])
        end_times.append(timestamps[-1])
        coverage_times.append(result_dict['coverage_time'][name])

        # Replace 'image.jpg' with the actual image file path for each person
        image_path = "scripts/fr_model/thumbnails/"+f"{name}.jpg"
        print(image_path)
        base64_data = image_to_base64(image_path)
        base64_images.append(base64_data)

    # Create a dictionary with the lists
    person_data = {
        'name': names,
        'startTime': start_times,
        'endTime': end_times,
        'coverageTime': coverage_times,
        'thumbnail': base64_images
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(person_data)

    # Display the DataFrame
    # print(df)
    json_data = df.to_json(orient='records')
    # Write the data to the JSON file
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)
    return json_data



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
