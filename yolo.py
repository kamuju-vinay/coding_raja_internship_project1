import torch
import cv2
from google.colab import files
from base64 import b64encode
from IPython.display import HTML
from google.colab.patches import cv2_imshow

# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Define paths
input_video_path = '/content/drive/MyDrive/datasets/download.mp4'
output_video_path = '/content/drive/MyDrive/datasets/output5.mp4'

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to process the video
def process_video(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")

        # Perform inference with YOLOv5
        results = model(frame)

        # Render the annotated frame with YOLOv5 predictions
        annotated_frame = results.render()[0]

        # Convert annotated_frame to BGR format for OpenCV
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write frame to output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process the video
process_video(input_video_path, output_video_path)


