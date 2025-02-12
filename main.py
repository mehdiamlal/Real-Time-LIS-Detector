import pickle
import cv2
import mediapipe as mp
import numpy as np
from data_collector import collect_data
from landmark_extractor import extract_landmarks
from train import train_model

model = pickle.load(open("./model.p", "rb"))["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

while True:
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        tmp = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                tmp.append(x)
                tmp.append(y)
                tmp.append(z)
                
                x_abs, y_abs = int(x * frame.shape[1]), int(y * frame.shape[0])
                x_min, y_min = min(x_min, x_abs), min(y_min, y_abs)
                x_max, y_max = max(x_max, x_abs), max(y_max, y_abs)
            
            prediction = model.predict([np.array(tmp)])
            predicted_letter = letters[int(prediction[0])].upper()
            
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.putText(frame, predicted_letter, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #print(predicted_letter)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
