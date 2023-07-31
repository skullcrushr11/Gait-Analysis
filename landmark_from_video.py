import cv2
import mediapipe as mp

# Function to draw pose landmarks on a frame
def draw_pose_landmarks(frame, landmarks):
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    # Draw lines between keypoints to represent skeleton
    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

# Main function to process the video
def process_video(input_path, output_path):
    mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Holistic
        results = mp_holistic.process(rgb_frame)

        # Draw pose landmarks on the frame
        draw_pose_landmarks(frame, results.pose_landmarks)

        out.write(frame)

        cv2.imshow('Video with Body Pose Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_path = 'istockphoto-811243290-640_adpp_is.mp4'
    output_video_path = 'output/result.mp4'
    process_video(input_video_path, output_video_path)