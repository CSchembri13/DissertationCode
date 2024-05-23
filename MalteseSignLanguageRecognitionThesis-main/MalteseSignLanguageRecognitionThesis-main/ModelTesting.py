import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score

from scipy import stats

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MalteseSignLanguageRecognitionThesis-main\largedata')

actions = np.array(['Kont', 'Dar', 'Flus', 'Missier', 'Passaport','Jekk','Widnejn','Issib','Ħalq','Tiegħi'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30,1662)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

try:
    model.load_weights('MalteseSignLanguageRecognitionThesis-main\MalteseSignLanguageRecognitionThesis-main\model1.h5')
except Exception as e:
    print(f"An error occurred: {e}")
#model.load_weights('model3.h5')

colors = [
    (245,117,16), 
    (117,245,16), 
    (16,117,245), 
    (190,37,187), 
    (216,213,51), 
    (230,146,39),
    (0,128,128),    # Teal
    (255,105,180),  # Hot Pink
    (75,0,130),     # Indigo
    (255,69,0),     # Orange Red
    (154,205,50)    # Yellow Green
]

# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         print(num)
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        
#     return output_frame


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    # Convert OpenCV image to PIL image
    pil_im = Image.fromarray(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(pil_im)
    font_path = "MalteseSignLanguageRecognitionThesis-main\MalteseSignLanguageRecognitionThesis-main\AbrilFatface-Regular.ttf"  # Replace with the path to a font that supports Maltese characters
    font = ImageFont.truetype(font_path, 32)  # Adjust font size as needed

    # Define the starting position for the rectangles and text
    y0, dy = 60, 40

    # First draw rectangles
    for num, prob in enumerate(res):
        # Ensure the color index is within the bounds
        color = colors[num % len(colors)]
        
        # Draw rectangle (Pillow uses RGB, OpenCV uses BGR)
        rect_start = (0, y0 + num * dy)
        rect_end = (int(prob * 100), y0 + num * dy + 30)
        draw.rectangle([rect_start, rect_end], fill=color[::-1])  # Reverse color for RGB

    # Then draw text on top of rectangles
    for num, action in enumerate(actions):
        text_position = (0, y0 + num * dy)
        draw.text(text_position, action, font=font, fill=(0, 0, 0))

    # Convert PIL image back to OpenCV image
    output_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return output_frame

plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

from os import environ

from google.cloud import translate


PROJECT_ID = environ.get("PROJECT_ID", "my-project-14198-1690529146451")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
        source_language_code='mt'
    )

    return response.translations[0]

def translateToEnglish(text):
    target_languages = ["en"]
    sentence = ' '.join(text)
    print(f" {sentence} ".center(50, "-"))
    for target_language in target_languages:
        translation = translate_text(sentence, target_language)
        source_language = translation.detected_language_code
        translated_text = translation.translated_text
        print(f"{source_language} → {target_language} : {translated_text}")

        # Set properties for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        line_type = 2
        
        # Create an image canvas
        canvas_height = 400
        image_width = 200
        canvas = np.zeros((canvas_height, 1000, 3), dtype=np.uint8)
        
        # Display original text
        original_text_size = cv2.getTextSize(sentence, font, font_scale, line_type)[0]
        original_text_x = (canvas.shape[1] - original_text_size[0]) // 2
        original_text_y = 50  # Display the original text at y=50 px
        cv2.putText(canvas, sentence, (original_text_x, original_text_y), font, font_scale, font_color, line_type)
        
        # Display translated text
        display_text = translated_text
        translated_text_size = cv2.getTextSize(display_text, font, font_scale, line_type)[0]
        translated_text_x = (canvas.shape[1] - translated_text_size[0]) // 2
        translated_text_y = original_text_y + 50  # Display the translated text below the original text
        cv2.putText(canvas, display_text, (translated_text_x, translated_text_y), font, font_scale, font_color, line_type)
        
        images = []  # List of images
        words = text
        for word in words:
            if (word == "Tiegħi"):
                image_path = f"MalteseSignLanguageRecognitionThesis-main/images/Tieghi.jpg"
            elif (word == "Ħalq"):
                image_path = f"MalteseSignLanguageRecognitionThesis-main/images/Halq.jpg"
            else:
                image_path = f"MalteseSignLanguageRecognitionThesis-main/images/{word}.jpg"
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.resize(img, (image_width, image_width))  # Resize to a fixed size
                images.append(img)

        # Positioning images
        start_x = 100  # Starting x position to draw images
        for img in images:
            canvas[translated_text_y + 30:translated_text_y + 30 + image_width, start_x:start_x + image_width] = img
            start_x += image_width + 10  # Move to the right for the next image

        # Display the image in a window
        cv2.imshow('Translation and Images', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

try:
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            if image is None or results is None:
                print("Failed to make detections")
                continue
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            if keypoints is None:
                print("Failed to extract keypoints")
                continue

            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

            #3. Viz logic
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                translateToEnglish(sentence)
                        else:
                            sentence.append(actions[np.argmax(res)])
                            translateToEnglish(sentence)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")
