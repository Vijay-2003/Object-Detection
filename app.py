import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

# Load the model and class labels
config_file = 'C:\\Users\\Dell\\Downloads\\54a8e8b51beb3bd3f770b79e56927bd7-2a20064a9d33b893dd95d2567da126d0ecd03e85\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'C:\\Users\\Dell\\Downloads\\54a8e8b51beb3bd3f770b79e56927bd7-2a20064a9d33b893dd95d2567da126d0ecd03e85\\ssd_mobilenet_v3_large_coco_2020_01_14\\frozen_inference_graph.pb'
file_name = 'C:\\Users\\Dell\\Downloads\\54a8e8b51beb3bd3f770b79e56927bd7-2a20064a9d33b893dd95d2567da126d0ecd03e85\\coco.txt'

try:
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
except Exception as e:
    st.error(f"Error loading model: {e}")

with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


def detect_objects(image):
    ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.5)
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if ClassInd <= 80:
            cv2.rectangle(image, boxes, (255, 0, 0), 2)
            cv2.putText(image, classlabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0), thickness=3)
    return image


st.title("Object Detection App")

st.sidebar.title("Upload Options")

upload_option = st.sidebar.selectbox("Choose an option", ("Image", "Video", "Webcam"))

if upload_option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting objects...")
        detected_image = detect_objects(image)
        st.image(detected_image, caption='Detected Image', use_column_width=True)

elif upload_option == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov', 'mkv'])
    if uploaded_file is not None:
        st.write("Detecting objects in video...")
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame)
            st.image(frame, channels="BGR")
        cap.release()

elif upload_option == "Webcam":
    st.write("Using webcam for real-time detection...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame)
            st.image(frame, channels="BGR")
    cap.release()
