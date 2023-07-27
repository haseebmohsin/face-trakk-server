import cv2

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

# Replace 'path_to_your_video_file.mp4' with the actual path to your video file
video_path = r"D:\Demo Video\headline.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the video time
    hours, minutes, seconds = get_video_time(cap)
    video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Add the video time to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)  # White color (BGR format)
    thickness = 2
    cv2.putText(frame, video_time, bottom_left_corner, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Show the frame with the video time
    cv2.imshow('Video with Time', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
