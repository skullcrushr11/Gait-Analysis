import os
import cv2
import mediapipe as mp

# Load the Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def draw_pose_landmarks(image, results):
    # Draw pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def process_frames_with_holistic(frames_folder, fps=30):
    frame_files = sorted(os.listdir(frames_folder))
    frame_paths = [os.path.join(frames_folder, file) for file in frame_files]

    if not frame_paths:
        raise ValueError("No image frames found in the specified folder.")

    # Create window to display frames
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)

    # Initialize Mediapipe Holistic and mp_drawing
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe Holistic
            results = holistic.process(image_rgb)

            # Draw pose landmarks on the frame and print landmark values
            draw_pose_landmarks(frame, results)

            cv2.imshow("Frames", frame)

            # Press 'q' key to quit
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


# Example usage:
frames_folder = "fyc/00_1"  #path to folder containing silhouette frames
fps = 5  # Frames per second

process_frames_with_holistic(frames_folder, fps)