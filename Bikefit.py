import cv2
import mediapipe as mp
import math
import numpy as np

# Fungsi untuk menghitung sudut antara tiga titik
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Fungsi untuk menerapkan filter rata-rata pada data sudut
def apply_average_filter(data, prev_data, filter_factor):
    filtered_data = (data * filter_factor) + (prev_data * (1 - filter_factor))
    return filtered_data

# Fungsi untuk memberikan label "Sudah Fit" atau "Belum Fit"
def get_fit_label(elbow_angle, hip_angle, knee_angle, ankling_range):
    if 60 <= hip_angle <= 110 and 150 <= elbow_angle <= 160 and 65 <= knee_angle <= 145 and 115 <= ankling_range <= 180:
        return "Sudah Fit"
    else:
        return "Belum Fit"

# Mengatur sumber video sebagai webcam eksternal dengan indeks 1
cap = cv2.VideoCapture(0)

# Mendapatkan ukuran asli webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Setup instance mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Variabel untuk menyimpan sudut sebelumnya
prev_elbow_angle = 0
prev_hip_angle = 0
prev_knee_angle = 0
prev_ankling_range = 0

# Faktor filter rata-rata
filter_factor = 0.8

# Tambahkan variabel untuk menyimpan jumlah gambar yang telah diambil
num_images_taken = 0

# Fungsi untuk menyimpan gambar
def save_image(frame):
    global num_images_taken
    image_name = f"image_{num_images_taken}.jpg"
    cv2.imwrite(image_name, frame)
    print(f"Image saved as {image_name}")
    num_images_taken += 1

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            
            # Calculate angles
            elbow_angle = (calculate_angle(right_shoulder, right_elbow, right_wrist)-4.94)
            hip_angle = (calculate_angle(right_shoulder, right_hip, right_knee)-31.17)
            knee_angle = (calculate_angle(right_hip, right_knee, right_ankle)-19)
            ankling_range = (calculate_angle(right_knee, right_ankle, right_foot_index)-17.53)
            
            # Apply average filter to angles
            elbow_angle = apply_average_filter(elbow_angle, prev_elbow_angle, filter_factor)
            hip_angle = apply_average_filter(hip_angle, prev_hip_angle, filter_factor)
            knee_angle = apply_average_filter(knee_angle, prev_knee_angle, filter_factor)
            ankling_range = apply_average_filter(ankling_range, prev_ankling_range, filter_factor)
            
            # Update previous angles
            prev_elbow_angle = elbow_angle
            prev_hip_angle = hip_angle
            prev_knee_angle = knee_angle
            prev_ankling_range = ankling_range
            
            # Get fit label
            fit_label = get_fit_label(elbow_angle, hip_angle, knee_angle, ankling_range)
            
            # Visualize angles and fit label
            cv2.putText(image, "Elbow Angle: {:.2f}".format(elbow_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Hip Angle: {:.2f}".format(hip_angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Knee Angle: {:.2f}".format(knee_angle), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Ankling Range: {:.2f}".format(ankling_range), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Fit Label: {}".format(fit_label), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 122), 2, cv2.LINE_AA)
            
            # Add logic to highlight elbow angle, hip angle, knee angle, and ankling range if they're within the desired range
            if 150 <= elbow_angle <= 160:
                cv2.putText(image, "Elbow Angle: {:.2f}".format(elbow_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Elbow Angle: {:.2f}".format(elbow_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if 60 <= hip_angle <= 110:
                cv2.putText(image, "Hip Angle: {:.2f}".format(hip_angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Hip Angle: {:.2f}".format(hip_angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if 65 <= knee_angle <= 145:
                cv2.putText(image, "Knee Angle: {:.2f}".format(knee_angle), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Knee Angle: {:.2f}".format(knee_angle), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            if 115 <= ankling_range <= 180:
                cv2.putText(image, "Ankling Range: {:.2f}".format(ankling_range), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Ankling Range: {:.2f}".format(ankling_range), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        except:
            pass
        
        # Resize the image to a smaller size for display
        image = cv2.resize(image, (1280, 720))
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Bike Fitting', image)

        # Add logic to take an image when the 's' key is pressed
        key = cv2.waitKey(1)
        if key == ord('s'):
            save_image(image)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
