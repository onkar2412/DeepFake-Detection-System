import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Set paths for real and fake video folders
REAL_VIDEOS_PATH = r'D:\Celeb-DF\Celeb-real'
FAKE_VIDEOS_PATH = r'D:\Celeb-DF\Celeb-synthesis'

# Parameters
IMG_SIZE = 128  # Image size for resizing
BATCH_SIZE = 16
EPOCHS = 10

# Helper function to extract frames and calculate editing percentage
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: Unable to read frames from {video_path}")
        cap.release()
        return np.array(frames), 0.0

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    prev_frame = None
    frame_diffs = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray_frame)
                frame_diffs.append(np.mean(diff))
            prev_frame = gray_frame
        else:
            print(f"Warning: Unable to read frame at index {idx} in {video_path}")
    cap.release()

    # Calculate percentage of editing based on frame differences
    if frame_diffs:
        editing_percentage = (np.mean(frame_diffs) / 255.0) * 100
    else:
        editing_percentage = 0.0

    return np.array(frames), editing_percentage

# Load data with error handling and include editing percentage
def load_data(real_videos_path, fake_videos_path, num_frames=10):
    X, y, editing_stats = [], [], []

    # Load real videos
    for video in os.listdir(real_videos_path):
        video_path = os.path.join(real_videos_path, video)
        if not os.path.isfile(video_path):
            continue
        frames, edit_percent = extract_frames(video_path, num_frames)
        for frame in frames:
            X.append(frame)
            y.append(0)  # Label for real videos
        editing_stats.append(edit_percent)

    # Load fake videos
    for video in os.listdir(fake_videos_path):
        video_path = os.path.join(fake_videos_path, video)
        if not os.path.isfile(video_path):
            continue
        frames, edit_percent = extract_frames(video_path, num_frames)
        for frame in frames:
            X.append(frame)
            y.append(1)  # Label for fake videos
        editing_stats.append(edit_percent)

    print(f"Average editing percentage across all videos: {np.mean(editing_stats):.2f}%")

    return np.array(X), np.array(y)

# Preprocess data
def preprocess_data(X, y):
    X = X / 255.0  # Normalize pixel values
    y = np.array(y).astype(np.float32)  # Ensure labels are in float32
    return X, y

# Split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Build the model
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nEvaluation Result:")
    print(f"Accuracy on the test set that freames are fake is: {accuracy * 100:.2f}%")

# Main function
def main():
    X, y = load_data(REAL_VIDEOS_PATH, FAKE_VIDEOS_PATH)
    if X.size == 0 or y.size == 0:
        print("Error: No data loaded. Check paths and video files.")
        return

    X, y = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model((IMG_SIZE, IMG_SIZE, 3))

    print("Starting training...")
    history = train_model(model, X_train, y_train)

    print("\nTraining completed. Evaluating on the test set...")
    evaluate_model(model, X_test, y_test)

    # Save the model
    model.save('deepfake_detection_model.h5')

if __name__ == "__main__":
    main()