import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from output_image import draw_landmarks_on_image
import numpy
import threading

annotated_frame = None
frame_lock = threading.Lock()
MODEL_PATH = './model/hand_landmarker.task'

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

HandLandmarkConnections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

start_time = time.monotonic_ns()

'''def draw_circle(frame: numpy.ndarray, x_pixel: int, y_pixel: int):
    cv2.cir'''

def convert_to_pixel_coordinates(x, y, width, height):
    x_pixel = int(x * width)
    y_pixel = int(y * height)    

def print_result(result: HandLandmarkerResult,  output_image: mp.Image, timestamp_ms: int):
    global annotated_frame

    handedness = result.handedness
    landmarks = result.hand_landmarks
    image = output_image.numpy_view()
    frame_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw = frame_to_draw.copy()
    if handedness:
        for i, h in enumerate(handedness):
            hand = h[0]
            if hand.score >= 0.5:
                print('Confidence Score:', hand.score)
                print(hand.category_name,"hand detected")
                width = output_image.width
                height = output_image.height

                for connections in HandLandmarkConnections:
                    start_co = landmarks[i][connections.start]
                    end_co = landmarks[i][connections.end]
                    start = (int(start_co.x * width), int(start_co.y * height))
                    end = (int(end_co.x * width), int(end_co.y * height))

                    cv2.line(draw, start, end, (0, 255, 0), 2)

                for j in range(21):
                    landmark = landmarks[i][j]
                    x_pixel = int(landmark.x * width)
                    y_pixel = int(landmark.y * height)
                    cv2.circle(draw, center=(x_pixel, y_pixel), radius=5, color=(0,0,255), thickness=-1)

    with frame_lock:
        # Draw full hand connections
        annotated_frame = draw
        

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not access the webcamera")
else:
    print("Webcam accessed successfully!")

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                success, frame = cap.read()
                if success:
                    frame_timestamp_ms = (time.monotonic_ns() -start_time) // 1_000_000
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    landmarker.detect_async(mp_image, frame_timestamp_ms)
                    with frame_lock:
                        new_frame = annotated_frame
                    if new_frame is not None:
                        cv2.imshow("Captured frame", new_frame)
                    else:
                        cv2.imshow('Captured frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Error: Could not capture a frame.")
        finally:
            cap.release()
            cv2.destroyAllWindows()