import random
import cv2
import mediapipe as mp
import os
import csv

SOURCE_DIRECTORY = "./LIS-fingerspelling-dataset/"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

features = []
labels = []
collected = 0
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

def write_to_csv(x, y, output_filename="landmarks.csv"):
    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")
    
    combined = [sublist + [num] for sublist, num in zip(x, y)]
    random.shuffle(combined)
    
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        for row in combined:
            writer.writerow(row)
    
    print(f"Landmarks saved successfully.")


for letter in os.listdir(SOURCE_DIRECTORY):
    if os.path.isdir(os.path.join(SOURCE_DIRECTORY, letter)):
        letter_path = os.path.join(SOURCE_DIRECTORY, letter)
        
        for img_filename in os.listdir(letter_path):
            img_path = os.path.join(letter_path, img_filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image {img_path}.")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            tmp = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z
                        tmp.append(x)
                        tmp.append(y)
                        tmp.append(z)

                features.append(tmp)
                labels.append(letters.index(letter))
        
        collected += 1
        print(f"Landmark collected for letter {letter.upper()}, {22 - collected} more left.")

write_to_csv(features, labels)