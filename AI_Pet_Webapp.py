import streamlit as st
import cv2
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from deepface import DeepFace
import time

# Load models and utilities outside of the function calls to avoid reloading them repeatedly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def capture_and_save_owner_embedding():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to RGB and detect faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        boxes, _ = mtcnn.detect(pil_img)
        
        if boxes is not None:
            # Extract the face with the largest area (most likely the main subject)
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_face_index = areas.index(max(areas))
            largest_face = mtcnn.extract(pil_img, [boxes[largest_face_index]], None)
            
            # Save the embedding of the largest detected face
            owner_embedding = resnet(largest_face).detach()
            torch.save(owner_embedding, 'owner_embedding.pt')
            
            print("Owner's face embedding captured and saved.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return owner_embedding

def is_owner_present():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    owner_embedding = torch.load('owner_embedding.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    owner_detected = False  # Flag to indicate if the owner is detected
    start_time = time.time()

    try:
        while time.time() - start_time < 5:  # Check for 5 seconds
            ret, frame = cap.read()
            if not ret:
                continue  # Skip this loop if frame is not captured correctly

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect faces
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                faces = mtcnn.extract(pil_img, boxes, None)
                embeddings = resnet(faces).detach()

                for embedding in embeddings:
                    # Calculate distance to the owner's embedding
                    distance = (embedding - owner_embedding).norm().item()
                    if distance < 0.6:  # threshold for recognition, tune based on your dataset
                        owner_detected = True
                        break

            if owner_detected:
                break  # Stop checking if owner is already detected

            # Display the frame (optional)
            cv_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow('Webcam', cv_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally: 
        cap.release()
        cv2.destroyAllWindows()

    return owner_detected

def continuous_emotion_detection():
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    emotions = []
    try:
        for _ in range(5):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Extract the face ROI (Region of Interest)
                    face_roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)  # Use original frame to get RGB ROI

                    # Perform emotion analysis on the face ROI
                    try:
                        # Perform emotion analysis on the face ROI
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                        # Determine the dominant emotion
                        emotion = result[0]['dominant_emotion']
                        emotions.append(emotion)
                    except Exception as e:
                        print(f"Error in emotion analysis: {e}")

        # Count occurrences of each emotion
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        # Find the emotion with the highest count
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return dominant_emotion

    finally:
        # Release the capture and close all windows
        cap.release()
        

def detect_emotion():
    # Call the continuous_emotion_detection function to get the dominant emotion
    dominant_emotion = continuous_emotion_detection()

    # Define responses based on the dominant emotion
    responses = {
        "sad": "Why are you sad? What happened?",
        "angry": "Did I do something wrong?",
        "happy": "Woof Woof! Tail Wag! I'm glad to see you too"
    }

    # Check if the dominant emotion is one of the keys in the responses dictionary
    if dominant_emotion in responses:
        # Return the appropriate response for the detected emotion
        bot_output = responses[dominant_emotion]


def chat(user_input):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    if user_input.lower() == 'quit':
        return "Woof woof! Bye!"

    # Encode user input and generate a response
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Ensure no previous context is included if it's leading to duplication of input in response
    chat_output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    
    # Decode the response
    bot_output = tokenizer.decode(chat_output_ids[0], skip_special_tokens=True)

    # Optionally strip the repeated input from the response
    if user_input.lower() in bot_output.lower():
        bot_output = bot_output[len(user_input):].strip()

    return bot_output


def main_page():
    st.markdown("# Woof Woof! I've Been Waiting to Play with You!")
    st.image("dog1.gif", use_column_width=True)
    chat_history = []

    # Text input for user message
    user_input = st.text_input("Type your message here:", key="user_input")
    send_button = st.button("Send", key="send_button")

    if send_button and user_input:
        bot_output = chat(user_input)
        chat_history.append(("You", user_input))
        chat_history.append(("Your Pet", bot_output))

        # Check for emotion detection opportunity
        if len(chat_history) >= 5:
            emotion_response = detect_emotion()
            chat_history.append(("Your Pet", emotion_response))

    # Display chat history
    if chat_history:
        for speaker, message in reversed(chat_history):
            st.text(f"{speaker}: {message}")

    # Sidebar to display chat history
    st.sidebar.markdown("# Chat History")
    for speaker, message in chat_history:
        st.sidebar.text(f"{speaker}: {message}")


def main():
    # Initially check if the owner is present
    owner_present = is_owner_present()

    if owner_present:
        # Directly go to the main page if owner is recognized
        main_page()
    else:
        if st.button("Get to Know Me", key="get_to_know"):
            capture_and_save_owner_embedding()
            st.markdown("Now that we've met, let's chat!")
            if st.button("Lets Chat!", key="lets_chat"):
                main_page()
        else:
            st.markdown("# Sorry, I don't talk to strangers.")
            st.image("notouch.png")

if __name__ == "__main__":
    main()
