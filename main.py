import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
import torchvision.transforms as T
from torchvision import models
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# define the model
class ASLModel(nn.Module):
    def __init__(self, num_class, device):
        super(ASLModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_class)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x
    
# define the predictor
class ASLPredictor():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    def __init__(self, checkpoint_path, device):
        self.model = ASLModel(26, device, checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
    
    def predict(self, img):
        # predict based on the net
        output = self.model(img)
        predicted = torch.softmax(output,dim=1) 
        _, predicted = torch.max(predicted, 1) 
        predicted = predicted.data.cpu() 
        idx = predicted[0]
        return self.letters[idx]

# define the hand detector
class HandDetector():
    def __init__(self, margin=10, font_size=1, font_thickness=2, text_color=(255, 255, 255)):
        # define constants
        self.MARGIN = margin
        self.FONT_SIZE = font_size
        self.FONT_THICKNESS = font_thickness
        self.TEXT_COLOR = text_color

        # define the hand detector
        base_options = python.BaseOptions(model_asset_path='./assets/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    # source: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            min_x = int(min(x_coordinates) * width) - self.MARGIN
            min_y = int(min(y_coordinates) * height) - self.MARGIN
            max_x = int(max(x_coordinates) * width) + self.MARGIN
            max_y = int(max(y_coordinates) * height) + self.MARGIN
            
            min_x = np.clip(min_x, 0, width)
            min_y = np.clip(min_y, 0, height)
            max_x = np.clip(max_x, 0, width)
            max_y = np.clip(max_y, 0, height)
            
            cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (min_x, min_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)
        return annotated_image
    
    def detect(self, image):
        # detect hand landmarks
        detection_result = self.detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        return detection_result, annotated_image


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    detector = HandDetector()
    while(True):
        ret, frame = vid.read()
        cv2.imwrite("./images/cache.jpg", frame)
        image = mp.Image.create_from_file("./images/cache.jpg")
        detection_result, annotated_frame = detector.detect(image)
        cv2.imshow('Real-Time-ASL', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()