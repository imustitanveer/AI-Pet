import streamlit as st
import cv2
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from deepface import DeepFace
import time


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

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                continue  # If no frame is captured, skip to the next iteration

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
                    yield emotion  # Yield the dominant emotion detected
                except Exception as e:
                    print(f"Error in emotion analysis: {e}")

    finally:
        # Release the capture and close all windows
        cap.release()
        

def chat(user_input, chat_history_ids=None):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Chatbot initialization message
    bot_output = ""
    
    # Continuous emotion detection generator
    emotions = continuous_emotion_detection()

    # Counter to track the number of emotions detected
    emotion_count = 0

    while True:
        # Get the next emotion from the generator
        emotion = next(emotions, None)

        if emotion:
            if emotion == "sad":
                bot_output = "Bot: Why are you sad? What happened?"
            elif emotion == "angry":
                bot_output = "Bot: Did I do something wrong?"

            # Increment emotion count
            emotion_count += 1

        # Check if emotions have been detected for 100 times
        if emotion_count >= 20:
            break

    if user_input.lower() == 'quit':
        bot_output = "Woof woof! Bye!"
    else:
        # Custom responses for pet-like behavior
        if "what are you" in user_input.lower():
            bot_output = "Bot: I'm your friendly chat pet! I like pats and treats!"
        elif "do you like" in user_input.lower():
            bot_output = "Bot: I love everything about you!"
        else:
            # Encode and generate response
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=1) if chat_history_ids is not None else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_output, chat_history_ids




# Define the welcome page
def welcome_page():
    st.markdown("""
                # Hello, Do I Know You?
                """)
    owner_present = is_owner_present()
    if not owner_present:
        if st.button("Get to Know Me"):
            # Run the function to capture and save owner embedding
            capture_and_save_owner_embedding()
            # Reload the page to check if owner is now present
            st.experimental_rerun()


def main_page():
    st.markdown("""
                # Welcome Back! I've Been Waiting to Play with You!
                """)
    st.image("dog1.gif", use_column_width=True)
    
    # Initialize chat history list
    chat_history = []
    user_input = st.text_input("You:")
    bot_response_slot = st.empty()
    
    # Initialize chatbot
    bot_output = "Woof! I'm your chat pet. Type 'quit' to stop playing with me."
    bot_response_slot.write(bot_output)
    
    # If user clicks enter or send button
    if st.button("Send"):
        # Call the chat function with user input
        bot_output, _ = chat(user_input)
        # Display bot's response above input box
        bot_response_slot.write(bot_output)
        # Append user input and bot response to chat history
        chat_history.append(("You:", user_input))
        chat_history.append(("Your Pet:", bot_output))
    
    # Display chat history above input box
    for speaker, message in chat_history:
        st.write(f"{speaker} {message}")


# Main function to run the Streamlit app
def main():
    # Check if owner is present
    owner_present = is_owner_present()
    if owner_present:
        main_page()
    else:
        welcome_page()

# Run the app
if __name__ == '__main__':
    main()