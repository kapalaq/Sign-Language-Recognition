import cv2 as cv
import numpy as np
import mediapipe as mp
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

from collections import namedtuple


# MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def recognizer(frame: np.ndarray, holistic: mp_holistic.Holistic) -> namedtuple:
    """
    Recognizing hands, face and body function
    :param frame: - image in numpy array format
    :param holistic: - Mediapipe built-in holistic solution object
    :return result: - coordinates of detected face, hands and body
    """
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = holistic.process(image)
    image.flags.writeable = True
    return result

def draw_all_landmarks(image: np.ndarray, landmarks: namedtuple):
    """
    Draw landmarks on image
    :param image: - image in numpy array format
    :param landmarks: - coordinates of detected face, hands and body
    :return: - None
    """
    mp_draw.draw_landmarks(image, landmarks.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                           mp_draw.DrawingSpec(thickness=1,
                                               circle_radius=1),
                           mp_draw.DrawingSpec(thickness=1,
                                               circle_radius=1))
    mp_draw.draw_landmarks(image, landmarks.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, landmarks.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(image, landmarks.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def get_coordinates(result: tuple) -> np.ndarray:
    """
    Convert coordinates into numpy array
    :param result: - coordinates of detected face, hands and body
    :return: - coordinates in numpy array format of shape=(1662,)
    """
    pose = np.zeros(132)
    i = 0
    if result.pose_landmarks is not None:
        for el in result.pose_landmarks.landmark:
            pose[i] = el.x
            pose[i + 1] = el.y
            pose[i + 2] = el.z
            pose[i + 3] = el.visibility
            i += 4
    left_hand = np.zeros(63)
    i = 0
    if result.left_hand_landmarks is not None:
        for el in result.left_hand_landmarks.landmark:
            left_hand[i] = el.x
            left_hand[i + 1] = el.y
            left_hand[i + 2] = el.z
            i += 3
    right_hand = np.zeros(63)
    i = 0
    if result.right_hand_landmarks is not None:
        for el in result.right_hand_landmarks.landmark:
            right_hand[i] = el.x
            right_hand[i + 1] = el.y
            right_hand[i + 2] = el.z
            i += 3
    face = np.zeros(1404)
    i = 0
    if result.face_landmarks is not None:
        for el in result.face_landmarks.landmark:
            face[i] = el.x
            face[i + 1] = el.y
            face[i + 2] = el.z
            i += 3
    return np.concatenate((pose, left_hand, right_hand, face))

def probability_graph(result: np.ndarray, targets: np.ndarray, frame: np.ndarray, colors: tuple):
    """
    Drawing probability distribution over all possible classes on frame
    :param result: - array of predicted probabilities
    :param targets: - array of actual values
    :param frame: - image in numpy array format
    :param colors: - colors for each class
    :return frame: - image with histogram of predicted probabilities
    """
    output_frame = frame.copy()
    for num, prob in enumerate(result):
        cv.rectangle(output_frame, (0, 55 + num * 38), (int(prob * 100), 75 + num * 38), colors[num], -1)
        cv.putText(output_frame, targets[num], (0, 70 + num * 38), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1, cv.LINE_AA)
    return output_frame


if __name__ == '__main__':
    # Constants
    targets = np.array(["hello", "thank you", "name", "country", "time",
                        "good", "morning", "afternoon", "night", "day", "nothing"])
    duration = 30
    coors_per_frame = 1662

    # Tensorflow Model initialization
    model = Sequential([
        Input(shape=(duration, coors_per_frame)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(256, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(128, activation='elu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='elu'),
        Dense(targets.size, activation='softmax'),
    ])
    model.load_weights("saved_models/HandSignLanuguageDetection.keras")

    # Model Running
    colors = (
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 128),  # Gray
        (16, 16, 16)  # Almost Black
    )
    snippet = list()
    history = list()
    predictions = list()
    threshold = 0.8

    # Set Mediapipe Model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        webcam = cv.VideoCapture(0)
        h, w = int(webcam.get(3)), int(webcam.get(4))
        print(h, w)
        try:
            while webcam.isOpened():
                ret, frame = webcam.read()

                results = recognizer(frame, holistic)
                draw_all_landmarks(frame, results)

                coors = get_coordinates(results)
                snippet.append(coors)
                snippet = snippet[-30:]

                if len(snippet) == 30:
                    result = model.predict(np.array(snippet).reshape(1, 30, 1662))[0]
                    predictions.append(np.argmax(result))

                    if np.unique(predictions[-10:])[0] == np.argmax(result) and np.unique(predictions[-10:]).size == 1:
                        if result[np.argmax(result)] > threshold:
                            if targets[np.argmax(result)] != "nothing":
                                if len(history) > 0:
                                    if targets[np.argmax(result)] != history[-1]:
                                        history.append(targets[np.argmax(result)])
                                else:
                                    history.append(targets[np.argmax(result)])

                    if len(history) > 4:
                        history = history[-4:]

                    frame = probability_graph(result, targets, frame, colors)

                cv.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv.putText(frame, ' '.join(history), (3, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(frame, "Press \"q\" to leave.", (w - 200, h - 180),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

                # Show to screen
                cv.imshow('webcam', frame)

                # Break gracefully
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            webcam.release()
            cv.destroyAllWindows()
        except Exception as e:
            print(e)
            webcam.release()
            cv.destroyAllWindows()