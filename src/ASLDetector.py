import cv2
import numpy as np
import mediapipe as mp

# define the hand detector
class ASLDetector():
    def __init__(self, predictor, margin=20, font_size=1, font_thickness=1, text_color=(76, 175, 80)):
        # define constants
        self.MARGIN = margin
        self.FONT_SIZE = font_size
        self.FONT_THICKNESS = font_thickness
        self.TEXT_COLOR = text_color
        
        # define the hand detector
        self.detector = mp.solutions.hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.75, max_num_hands=2)
        self.predictor = predictor

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)
        hand_landmarks_list = detection_result.multi_hand_landmarks
        if hand_landmarks_list:
            # for each detected hand
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx].landmark
            
                # Get the top left and bottom right corner of the detected hand's bounding box.
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
                
                hand_frame = rgb_image[min_y:max_y, min_x:max_x]
                letter = self.predictor.predict(hand_frame)
                annotated_image = cv2.putText(annotated_image, letter, (100,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, self.TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)
                annotated_image = cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), self.TEXT_COLOR, 2)
        return annotated_image
    
    def detect(self, image):
        # detect hand landmarks
        detection_result = self.detector.process(image)
        annotated_image = self.draw_landmarks_on_image(image, detection_result)
        return annotated_image
