# # # import cv2
# # # from deepface import DeepFace

# # # # Load face cascade classifier
# # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # # Start capturing video
# # # cap = cv2.VideoCapture(0)

# # # while True:
# # #     # Capture frame-by-frame
# # #     ret, frame = cap.read()

# # #     # Convert frame to grayscale
# # #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # #     # Convert grayscale frame to RGB format
# # #     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# # #     # Detect faces in the frame
# # #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # #     for (x, y, w, h) in faces:
# # #         # Extract the face ROI (Region of Interest)
# # #         face_roi = rgb_frame[y:y + h, x:x + w]

        
# # #         # Perform emotion analysis on the face ROI
# # #         result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

# # #         # Determine the dominant emotion
# # #         emotion = result[0]['dominant_emotion']

# # #         # Draw rectangle around face and label with predicted emotion
# # #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# # #         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# # #     # Display the resulting frame
# # #     cv2.imshow('Real-time Emotion Detection', frame)

# # #     # Press 'q' to exit
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # # Release the capture and close all windows
# # # cap.release()
# # # cv2.destroyAllWindows()

# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # Detect emotion from live. cam capture and print precentage results
# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # import cv2
# # from deepface import DeepFace

# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()

# #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     for (x, y, w, h) in faces:
# #         face_roi = rgb_frame[y:y + h, x:x + w]

# #         result = DeepFace.analyze(face_roi, actions=['emotion','gender','age'], enforce_detection=False)
        
# #         emotion_data = result[0]['emotion']
        
# #         print("Emotion percentages:")
# #         for emotion, percentage in emotion_data.items():
# #             print(f"{emotion}: {percentage:.2f}%")
        
# #         emotion = result[0]['dominant_emotion']
# #         gender = result[0]['dominant_gender']
# #         age = result[0]['age']
        
        
# #         print(f"gender: {gender}")
# #         print(f"age: {age}")
        
        
# #         label = f"{emotion}, {gender}, {age}"

# #         # Draw rectangle around face and label with predicted emotion
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# #     cv2.imshow('Real-time Emotion Detection', frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the capture and close all windows
# # cap.release()
# # cv2.destroyAllWindows()


# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # Detect emotion from image input/
# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# # import cv2
# # from deepface import DeepFace

# # # Load face cascade classifier
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # Load the input image from a file
# # image_path = '/Users/macbook/Desktop/H_Detection/train/Fear/images (1).jpg'  # Replace with the path to your image
# # frame = cv2.imread(image_path)

# # # Convert frame to grayscale
# # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # # Convert grayscale frame to RGB format
# # rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# # # Detect faces in the image
# # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# # for (x, y, w, h) in faces:
# #     # Extract the face ROI (Region of Interest)
# #     face_roi = rgb_frame[y:y + h, x:x + w]

# #     # Perform emotion, age, gender, and race analysis on the face ROI
# #     result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    
# #     # result = DeepFace.analyze(face_roi, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
    
    
# #     # Extract the emotion percentages
# #     emotion_data = result[0]['emotion']
    
# #     # Print emotion percentages
    
# #     print("Emotion percentages:")
# #     for emotion, percentage in emotion_data.items():
# #         print(f"{emotion}: {percentage:.2f}%")
    
# #     # Print age, gender, and race
# #     emotion = result[0]['dominant_emotion']
    
# #     # age = result[0]['age']
# #     # gender = result[0]['dominant_gender']
# #     # race = result[0]['dominant_race']
    
# #     print(f"\nemotion of human : {emotion}\n")
    
# #     # print(f"\nAge: {age}\n")
# #     # print(f"Gender: {gender}")
# #     # print(f"Race: {race}")

# #     # Determine the dominant emotion
# #     emotion = result[0]['dominant_emotion']

# #     # Draw rectangle around face and label with predicted emotion, age, gender, and race
# #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
# #     # label = f"{emotion}, {gender}, {age}, {race}"
    
# #     label = f"{emotion}"
    
# #     cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# # # Display the resulting image with detected faces and emotions, age, gender, and race
# # cv2.imshow('Emotion, Age, Gender, Race Detection', frame)

# # # Wait until a key is pressed, then close the window
# # cv2.waitKey(0)

# # # Close all OpenCV windows
# # cv2.destroyAllWindows()


# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # Detect emotion,age and gender from image video/
# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# # import cv2
# # from deepface import DeepFace
# # import json

# # # Load the pre-trained face detection model
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # Start capturing video from the webcam
# # cap = cv2.VideoCapture(0)

# # while True:
# #     # Read a frame from the video
# #     ret, frame = cap.read()

# #     # Convert the frame to grayscale for face detection
# #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     # Convert the grayscale frame back to RGB for DeepFace analysis
# #     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# #     # Detect faces in the grayscale frame
# #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     # Loop through the detected faces
# #     for (x, y, w, h) in faces:
# #         # Extract the region of interest (ROI) where the face is located
# #         face_roi = rgb_frame[y:y + h, x:x + w]

# #         # Use DeepFace to analyze the face for emotion, gender, and age
# #         result = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)
        
# #         # Extract emotion, gender, and age data
# #         emotion_data = result[0]['emotion']
# #         dominant_emotion = result[0]['dominant_emotion']
# #         gender_dominant = result[0]['dominant_gender']
# #         gender = result[0]['gender']
        
# #         age = result[0]['age']

# #         # Prepare the data in JSON format
# #         data = {
# #             "emotion_percentages": emotion_data,
# #             "dominant_emotion": dominant_emotion,
# #             "gender_dominant": gender_dominant,
# #             "age": age,
# #             "gender": gender
# #         }

# #         # Print the JSON data
# #         json_data = json.dumps(data, indent=5)
# #         print(json_data)

# #         # Create a label to display on the video feed
# #         label = f"{dominant_emotion}, {gender_dominant}, {age}"

# #         # Draw a rectangle around the face and put the label on the frame
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# #     # Display the frame with the face and label
# #     cv2.imshow('Real-time Emotion Detection', frame)

# #     # Break the loop if the 'q' key is pressed
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the video capture and close all windows
# # cap.release()
# # cv2.destroyAllWindows()

# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # Detect emotion,age and gender from image video with mutiple faces/
# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # import cv2
# # from deepface import DeepFace
# # import json

# # # Load the pre-trained face detection model
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # Start capturing video from the webcam
# # cap = cv2.VideoCapture(0)

# # while True:
# #     # Read a frame from the video
# #     ret, frame = cap.read()

# #     # Convert the frame to grayscale for face detection
# #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     # Convert the grayscale frame back to RGB for DeepFace analysis
# #     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# #     # Detect faces in the grayscale frame
# #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     face_data_list = []  # List to hold JSON data for all detected faces

# #     # Loop through the detected faces
# #     for (x, y, w, h) in faces:
# #         # Extract the region of interest (ROI) where the face is located
# #         face_roi = rgb_frame[y:y + h, x:x + w]

# #         # Use DeepFace to analyze the face for emotion, gender, and age
# #         result = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)
        
# #         # Extract emotion, gender, and age data
# #         emotion_data = result[0]['emotion']
# #         dominant_emotion = result[0]['dominant_emotion']
# #         gender_dominant = result[0]['dominant_gender']
# #         gender = result[0]['gender']
# #         age = result[0]['age']

# #         # Prepare the data in JSON format
# #         data = {
# #             "emotion_percentages": emotion_data,
# #             "dominant_emotion": dominant_emotion,
# #             "gender_dominant": gender_dominant,
# #             "age": age,
# #             "gender": gender
# #         }

# #         # Append this face's data to the list
# #         face_data_list.append(data)

# #         # Create a label to display on the video feed for each face
# #         label = f"{dominant_emotion}, {gender_dominant}, {age}"

# #         # Draw a rectangle around the face and put the label on the frame
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# #     # Print the JSON data for all faces in the current frame
# #     if face_data_list:
# #         json_data = json.dumps(face_data_list, indent=4)
# #         print(json_data)

# #     # Display the frame with the faces and labels
# #     cv2.imshow('Real-time Emotion Detection', frame)

# #     # Break the loop if the 'q' key is pressed
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the video capture and close all windows
# # cap.release()
# # cv2.destroyAllWindows()

# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # print data with image in json of multiple images fcess
# # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# # import cv2
# # from deepface import DeepFace
# # import json
# # import base64
# # import numpy as np
# # from io import BytesIO
# # from PIL import Image

# # # Load the pre-trained face detection model
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # Start capturing video from the webcam
# # cap = cv2.VideoCapture(0)

# # def encode_image_to_base64(image):
# #     """Encode the given image (NumPy array) to a Base64 string."""
# #     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
# #     buffer = BytesIO()
# #     pil_image.save(buffer, format="JPEG")  # Save image to buffer as JPEG
# #     base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Encode buffer to Base64
# #     return base64_image

# # while True:
# #     # Read a frame from the video
# #     ret, frame = cap.read()

# #     # Convert the frame to grayscale for face detection
# #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     # Convert the grayscale frame back to RGB for DeepFace analysis
# #     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# #     # Detect faces in the grayscale frame
# #     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     face_data_list = []  # List to hold JSON data for all detected faces

# #     # Loop through the detected faces
# #     for (x, y, w, h) in faces:
# #         # Extract the region of interest (ROI) where the face is located
# #         face_roi = rgb_frame[y:y + h, x:x + w]

# #         # Use DeepFace to analyze the face for emotion, gender, and age
# #         result = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)
        
# #         # Extract emotion, gender, and age data
# #         emotion_data = {key: float(value) for key, value in result[0]['emotion'].items()}  # Convert to Python float
# #         dominant_emotion = str(result[0]['dominant_emotion'])  # Ensure string format
# #         gender_dominant = str(result[0]['dominant_gender'])
# #         age = int(result[0]['age'])  # Convert NumPy int to Python int
        
# #         # Capture and encode the face image to Base64
# #         face_image = frame[y:y + h, x:x + w]  # Capture the face area from the original frame
# #         base64_face_image = encode_image_to_base64(face_image)  # Encode image to Base64

# #         # Prepare the data in JSON format, including face bounding box and image
# #         data = {
# #             "bounding_box": {
# #                 "x": int(x),
# #                 "y": int(y),
# #                 "width": int(w),
# #                 "height": int(h)
# #             },
# #             "emotion_percentages": emotion_data,
# #             "dominant_emotion": dominant_emotion,
# #             "gender_dominant": gender_dominant,
# #             "age": age,
# #             # "face_image_base64": base64_face_image  # Add the Base64-encoded face image
# #         }

# #         # Append this face's data to the list
# #         face_data_list.append(data)

# #         # Create a label to display on the video feed for each face
# #         label = f"{dominant_emotion}, {gender_dominant}, {age}"

# #         # Draw a rectangle around the face and put the label on the frame
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# #         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# #     # Print the JSON data for all faces in the current frame
# #     if face_data_list:
# #         json_data = json.dumps(face_data_list, indent=4)
# #         print(json_data)

# #     # Display the frame with the faces and labels
# #     cv2.imshow('Real-time Emotion Detection', frame)

# #     # Break the loop if the 'q' key is pressed
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the video capture and close all windows
# # cap.release()
# # cv2.destroyAllWindows()

# # frame rate control

# import cv2
# from deepface import DeepFace
# import json
# import base64
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import time

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)

# def encode_image_to_base64(image):
#     """Encode the given image (NumPy array) to a Base64 string."""
#     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
#     buffer = BytesIO()
#     pil_image.save(buffer, format="JPEG")  # Save image to buffer as JPEG
#     base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Encode buffer to Base64
#     return base64_image

# frame_rate = 10  # Frames per second
# prev = time.time()

# while True:
#     time_elapsed = time.time() - prev
#     ret, frame = cap.read()
#     if time_elapsed > 1./frame_rate:
#         prev = time.time()
        
#         # frame = cv2.resize(frame, (640, 480))

#         # Convert the frame to grayscale for face detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Convert the grayscale frame back to RGB for DeepFace analysis
#         rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#         # Detect faces in the grayscale frame
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         face_data_list = []  # List to hold JSON data for all detected faces

#         # Loop through the detected faces
#         for (x, y, w, h) in faces:
#             # Extract the region of interest (ROI) where the face is located
#             face_roi = rgb_frame[y:y + h, x:x + w]

#             # Use DeepFace to analyze the face for emotion, gender, and age
#             result = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)

#             # Extract emotion, gender, and age data
#             emotion_data = {key: float(value) for key, value in result[0]['emotion'].items()}  # Convert to Python float
#             dominant_emotion = str(result[0]['dominant_emotion'])  # Ensure string format
#             gender_dominant = str(result[0]['dominant_gender'])
#             age = int(result[0]['age'])  # Convert NumPy int to Python int

#             # Capture and encode the face image to Base64
#             face_image = frame[y:y + h, x:x + w]  # Capture the face area from the original frame
#             base64_face_image = encode_image_to_base64(face_image)  # Encode image to Base64

#             # Prepare the data in JSON format, including face bounding box and image
#             data = {
#                 "bounding_box": {
#                     # "x": int(x),
#                     # "y": int(y),
#                     # "width": int(w),
#                     # "height": int(h)
#                 },
#                 "emotion_percentages": emotion_data,
#                 "dominant_emotion": dominant_emotion,
#                 "gender_dominant": gender_dominant,
#                 "age": age,
#                 # "face_image_base64": base64_face_image  # Add the Base64-encoded face image
#             }

#             # Append this face's data to the list
#             face_data_list.append(data)

#             # Create a label to display on the video feed for each face
#             label = f"{dominant_emotion}, {gender_dominant}, {age}"

#             # Draw a rectangle around the face and put the label on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#         # Print the JSON data for all faces in the current frame
#         if face_data_list:
#             json_data = json.dumps(face_data_list, indent=4)
#             print(json_data)

#         # Display the frame with the faces and labels
#         cv2.imshow('Real-time Emotion Detection', frame)

#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()



import cv2
from deepface import DeepFace
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import time
from flask import Flask, render_template, Response
from flask_cors import CORS

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create Flask app
app = Flask(__name__)
CORS(app)

# Frame rate for video
frame_rate = 10

def encode_image_to_base64(image):
    """Encode the given image (NumPy array) to a Base64 string."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")  # Save image to buffer as JPEG
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Encode buffer to Base64
    return base64_image

def generate_frames():
    """Generate video frames for Flask app, with emotion detection."""
    cap = cv2.VideoCapture(0)
    prev = time.time()

    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale frame back to RGB for DeepFace analysis
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            face_data_list = []  # List to hold JSON data for all detected faces

            # Loop through the detected faces
            for (x, y, w, h) in faces:
                # Extract the region of interest (ROI) where the face is located
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Use DeepFace to analyze the face for emotion, gender, and age
                result = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)

                # Extract emotion, gender, and age data
                emotion_data = {key: float(value) for key, value in result[0]['emotion'].items()}  # Convert to Python float
                dominant_emotion = str(result[0]['dominant_emotion'])  # Ensure string format
                gender_dominant = str(result[0]['dominant_gender'])
                age = int(result[0]['age'])  # Convert NumPy int to Python int

                # Prepare the data in JSON format
                data = {
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "emotion_percentages": emotion_data,
                    "dominant_emotion": dominant_emotion,
                    "gender_dominant": gender_dominant,
                    "age": age
                }

                # Append this face's data to the list
                face_data_list.append(data)

                # Create a label to display on the video feed for each face
                label = f"{dominant_emotion}, {gender_dominant}, {age}"

                # Draw a rectangle around the face and put the label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Print the JSON data for all faces in the current frame
            if face_data_list:
                json_data = json.dumps(face_data_list, indent=4)
                print(json_data)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of a multipart HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Render the main page with a link to start emotion detection."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Route to stream video frames to the web page."""
    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response



if __name__ == '__main__':
    app.run(debug=True)
