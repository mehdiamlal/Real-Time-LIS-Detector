# Real-Time-LIS-Detector
This project implements a real-time recognition system for **Italian Sign Language (LIS)** letters using **landmark extraction** and **machine learning models**, based on the research paper *Real-time Italian Sign Language Recognition with Deep Learning*.

## Features
- Uses **MediaPipe Hands** for extracting hand landmarks instead of raw image processing.
- **Random Forest Classifier** for sign classification instead of CNN/VGG19.
- Works with **live webcam input** for real-time sign detection.
- Generates a **lightweight model** that runs efficiently without deep learning frameworks.

## Project Structure
```
Real-Time-LIS-Detector
├── data_collector.py        # Collects additional dataset samples
├── landmark_extractor.py    # Extracts hand landmarks using MediaPipe
├── train.py                 # Trains and evaluates the model
├── main.py                  # Runs real-time predictions using a webcam
├── README.md                # Project Documentation
```

## Data Collection
Since the full dataset from the original paper was not available, a subset (6,428 images, 3 angles) was used. Additional samples were created using **`data_collector.py`**.

## Landmark Extraction
Instead of raw image classification, this project extracts **21 key hand landmarks (x, y, z coordinates)** using **MediaPipe Hands**. This reduces data complexity, allowing us to use traditional ML classifiers instead of CNNs.

Run the following command to extract landmarks from images:
```bash
python landmark_extractor.py
```
This will save landmarks to `landmarks.csv`.

## Model Training
The model is trained on extracted landmarks using a **Random Forest Classifier**.
```bash
python train.py
```
This script:
- Loads landmark data from `landmarks.csv`.
- Splits the dataset (80% train, 20% test).
- Trains and evaluates a **Random Forest Classifier**.
- Saves the trained model using `pickle`.

## Real-Time Prediction
Use your **webcam** to predict LIS letters in real-time.
```bash
python main.py
```
- **OpenCV** captures the hand in a bounding box.
- **MediaPipe Hands** extracts landmarks.
- The trained model predicts the LIS letter.
- The letter is displayed above the hand.

## Acknowledgments
- Inspired by *Real-time Italian Sign Language Recognition with Deep Learning*.
- Uses **MediaPipe Hands** for landmark extraction.