import os
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

import cv2
from flask import Flask, Response

import mediapipe as mp
# from mediapipe import solutions
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2

app = Flask(__name__)
video = cv2.VideoCapture(0)
os.makedirs("./images", exist_ok=True)

# class Timer():
#     def __init__(self, duration):
#         self.tik = time.time()
#         self.tok = time.time()
#         self.duration = duration
        
#     def times_up(self):
#         self.tok = time.time()
#         if self.tok - self.tik > self.duration:
#             self.tik = self.tok
#             return True
#         return False
    
# # define the model
# class ASLModel(nn.Module):
#     def __init__(self, num_class, device):
#         super(ASLModel, self).__init__()
#         self.model = models.resnet18(weights=None)
#         self.model.fc = nn.Linear(512, num_class)
#         self.model = self.model.to(device)

#     def forward(self, x):
#         x = self.model(x)
#         return x
    
# # define the predictor
# class ASLPredictor():
#     letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
#     def __init__(self, checkpoint_path, device):
#         self.model = ASLModel(24, device)
#         self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#         self.model.eval()
#         self.predicted = None
#         self.timer = Timer(1)
    
#     def predict(self, img):
#         if self.timer.times_up():
#             # transform the image to the correct format
#             # TODO: find a better way to do this
#             img = torch.from_numpy(img).float()
#             img = img.permute(2, 0, 1)
#             img = transforms.Resize((28, 28))(img)
#             img = img.unsqueeze(0)
#             img /= 255.0
            
#             # predict the letter
#             output = self.model(img)
#             predicted = torch.softmax(output,dim=1) 
#             _, predicted = torch.max(predicted, 1) 
#             predicted = predicted.cpu() 
#             idx = predicted[0]
#             self.predicted = self.letters[idx]
#         return self.predicted

# # define the hand detector
# class ASLDetector():
#     def __init__(self, predictor, margin=20, font_size=1, font_thickness=1, text_color=(255, 255, 255)):
#         # define constants
#         self.predictor = predictor
#         self.MARGIN = margin
#         self.FONT_SIZE = font_size
#         self.FONT_THICKNESS = font_thickness
#         self.TEXT_COLOR = text_color

#         # define the hand detector
#         base_options = python.BaseOptions(model_asset_path='./assets/hand_landmarker.task')
#         options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence= 0.7)
#         self.detector = vision.HandLandmarker.create_from_options(options)

#     # source: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
#     def draw_landmarks_on_image(self, rgb_image, detection_result):
#         hand_landmarks_list = detection_result.hand_landmarks
#         handedness_list = detection_result.handedness
#         annotated_image = np.copy(rgb_image)
#         # Loop through the detected hands to visualize.
#         for idx in range(len(hand_landmarks_list)):
#             hand_landmarks = hand_landmarks_list[idx]
#             handedness = handedness_list[idx]

#             # Draw the hand landmarks.
#             hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
#             # solutions.drawing_utils.draw_landmarks(
#             #     annotated_image,
#             #     hand_landmarks_proto,
#             #     solutions.hands.HAND_CONNECTIONS,
#             #     solutions.drawing_styles.get_default_hand_landmarks_style(),
#             #     solutions.drawing_styles.get_default_hand_connections_style()
#             # )

#             # Get the top left and bottom right corner of the detected hand's bounding box.
#             height, width, _ = annotated_image.shape
#             x_coordinates = [landmark.x for landmark in hand_landmarks]
#             y_coordinates = [landmark.y for landmark in hand_landmarks]
            
#             min_x = int(min(x_coordinates) * width) - self.MARGIN
#             min_y = int(min(y_coordinates) * height) - self.MARGIN
#             max_x = int(max(x_coordinates) * width) + self.MARGIN
#             max_y = int(max(y_coordinates) * height) + self.MARGIN
            
#             min_x = np.clip(min_x, 0, width)
#             min_y = np.clip(min_y, 0, height)
#             max_x = np.clip(max_x, 0, width)
#             max_y = np.clip(max_y, 0, height)
            
#             hand_frame = rgb_image[min_y:max_y, min_x:max_x]
#             letter = self.predictor.predict(hand_frame)
#             annotated_image = cv2.putText(annotated_image, letter, (100,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)
#             annotated_image = cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
#             cv2.imwrite("./images/hand_cache.jpg", hand_frame)
#         return annotated_image
    
#     def detect(self, image):
#         # detect hand landmarks
#         detection_result = self.detector.detect(image)
#         annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)
#         annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
#         return annotated_image


def gen(video):
    # predictor = ASLPredictor("./assets/model_3.pt", "cpu")
    # detector = ASLDetector(predictor=predictor)
    while True:
        success, image = video.read()
        # cv2.imwrite("./images/cache.jpg", frame)
        # image = mp.Image.create_from_file("./images/cache.jpg")
        
        # # detect the hand and display the result
        # annotated_frame = detector.detect(image)
        
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2207, threaded=False)