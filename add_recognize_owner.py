import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

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

# To Call 
# capture_and_save_owner_embedding()


def recognize_and_highlight():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    owner_embedding = torch.load('owner_embedding.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect faces
            boxes, _ = mtcnn.detect(pil_img)
            draw = ImageDraw.Draw(pil_img)
            if boxes is not None:
                faces = mtcnn.extract(pil_img, boxes, None)
                embeddings = resnet(faces).detach()

                for i, box in enumerate(boxes):
                    # Calculate distance to the owner's embedding
                    distance = (embeddings[i] - owner_embedding).norm().item()
                    if distance < 0.6:  # threshold for recognition
                        outline_color = (0, 255, 0)  # Green for owner
                    else:
                        outline_color = (255, 0, 0)  # Red for others

                    draw.rectangle(box.tolist(), outline=outline_color, width=6)

            # Display the image
            cv_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow('Webcam', cv_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        

# to highlight in webcam feed
# recognize_and_highlight()