import cv2


def main():
    input_path = "/home/ahmed/facenet-pytorch/testing_videos/test/Arif_Alvi2.mp4"
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames of the input video
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    print(input_path)
    print("input video total frames:", total_frames)
    print("input video FPS:", output_fps)

main()