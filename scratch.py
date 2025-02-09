import cv2
import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from playsound import playsound  # Import playsound for playing sound

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand detection
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define gesture labels
gestures = ['yes', 'no', 'hello', 'goodbye']
gesture_data = {
    'yes': [],
    'no': [],
    'hello': [],
    'goodbye': []
}

# Function to collect gesture data
def collect_gesture_data(label):
    data = []
    print(f"Collecting data for '{gestures[label]}' gesture...")

    for i in range(100):  # Collect 100 frames for each gesture
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a later mirror view
        frame = cv2.flip(frame, 1)

        # Process the frame for hand landmarks
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Normalize landmarks
                landmarks_array = []
                for landmark in landmarks.landmark:
                    landmarks_array.append([landmark.x, landmark.y, landmark.z])
                data.append(np.array(landmarks_array).flatten())  # Flatten to 1D array
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with hand landmarks
        cv2.imshow("Gesture Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Return the collected data
    return np.array(data)

# Collect data for each gesture
yes_data = collect_gesture_data(0)
no_data = collect_gesture_data(1)
hello_data = collect_gesture_data(2)
goodbye_data = collect_gesture_data(3)

# Save collected data and labels
def save_data(data, label):
    np.save(f"{gestures[label]}_data.npy", data)
    labels = np.array([label] * len(data))
    np.save(f"{gestures[label]}_labels.npy", labels)

save_data(yes_data, 0)
save_data(no_data, 1)
save_data(hello_data, 2)
save_data(goodbye_data, 3)

# Close the camera window
cap.release()
cv2.destroyAllWindows()

# Load gesture data
yes_data = np.load('yes_data.npy')
no_data = np.load('no_data.npy')
hello_data = np.load('hello_data.npy')
goodbye_data = np.load('goodbye_data.npy')

# Stack all the data and labels
X = np.vstack([yes_data, no_data, hello_data, goodbye_data])
y = np.hstack(
    [np.zeros(len(yes_data)), np.ones(len(no_data)), np.full(len(hello_data), 2), np.full(len(goodbye_data), 3)])

# Normalize the data
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Build the model
model = Sequential([
    Dense(128, input_shape=(21 * 3,), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')  # 4 classes (yes, no, hello, goodbye)
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Save the trained model using the recommended Keras format
model.save('gesture_recognition_model.keras')

# Function to play sound for recognized gesture
def play_sound(gesture):
    # Define sounds for each gesture (just placeholder paths, adjust as needed)
    sound_files = {
        'yes': "yes_sound.wav",
        'no': "no_sound.wav",
        'hello': "hello_sound.wav",
        'goodbye': "goodbye_sound.wav",
    }

    # Play corresponding sound file
    if gesture in sound_files:
        playsound(sound_files[gesture])

# Function to predict gesture in real-time
def predict_gesture(frame):
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            landmarks_array = []
            for landmark in landmarks.landmark:
                landmarks_array.append([landmark.x, landmark.y, landmark.z])
            data = np.array(landmarks_array).flatten().reshape(1, -1)
            data = data / np.linalg.norm(data)  # Normalize input

            prediction = model.predict(data)
            predicted_class = np.argmax(prediction)

            gesture = gestures[predicted_class]
            return gesture
    return None

# Start real-time gesture recognition
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        gesture = predict_gesture(frame)

        if gesture:
            print(f"Recognized Gesture: {gesture}")
            play_sound(gesture)  # Play sound for recognized gesture

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Gesture recognition interrupted.")

# Close the camera window
cap.release()
cv2.destroyAllWindows()
