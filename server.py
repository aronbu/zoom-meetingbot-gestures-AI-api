import csv
import itertools
from flask import Flask, request, jsonify
import cv2 as cv
import numpy as np
import mediapipe as mp

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from collections import deque, Counter
import copy

app = Flask(__name__)

# Initialize models



# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Coordinate history
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)


@app.route('/detect_gesture', methods=['POST'])
def detect_gesture():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    keypoint_classifier = KeyPointClassifier()
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id is not None:
                if hand_sign_id == "not used atm":  # Point gesture == 2
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                return jsonify({
                    'hand_sign': keypoint_classifier_labels[hand_sign_id],
                })
            else:
                point_history.append([0, 0])
    else:
        point_history.append([0, 0])

    return jsonify({
        'hand_sign': None,
        'finger_gesture': None
    })


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
