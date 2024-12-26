from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from flask_cors import CORS
import io
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import logging
import re

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load models
model_files = {
    "faceProto": "opencv_face_detector.pbtxt",
    "faceModel": "opencv_face_detector_uint8.pb",
    "ageProto": "age_deploy.prototxt",
    "ageModel": "age_net.caffemodel",
    "genderProto": "gender_deploy.prototxt",
    "genderModel": "gender_net.caffemodel"
}

try:
    faceNet = cv2.dnn.readNet(model_files["faceModel"], model_files["faceProto"])
    ageNet = cv2.dnn.readNet(model_files["ageModel"], model_files["ageProto"])
    genderNet = cv2.dnn.readNet(model_files["genderModel"], model_files["genderProto"])
except Exception as e:
    logging.error("Failed to load models: ", exc_info=True)
    raise RuntimeError("Model files could not be loaded. Please check the file paths and names.")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 20

def recommend_product_images_for_age_gender_with_fallback(age, gender, data_path):
    """
    Recommends product images based on age and gender with fallback mechanism.
    
    Args:
        age (int): Age of the user.
        gender (str): Gender of the user ('Male' or 'Female').
        data_path (str): Path to the CSV file containing product data.
    
    Returns:
        list: List of dictionaries containing product name and image URL.
    """
    df = pd.read_csv(data_path)
    filtered_df = df[(df['Age'] == age) & (df['Gender'].str.upper() == gender.upper())]

    if filtered_df.empty:
        # Expand search criteria if no exact matches
        age_min = max(1, age - 5)
        age_max = age + 5
        filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max) & (df['Gender'].str.upper() == gender.upper())]

    if filtered_df.empty:
        return [], []

    # Compute weighted ratings
    user_item_matrix = filtered_df.pivot_table(index='User ID', columns='Product Name', values='Rating', fill_value=0)
    sparse_user_item = sparse.csr_matrix(user_item_matrix.values)
    similarities = cosine_similarity(sparse_user_item)
    similarities_df = pd.DataFrame(similarities, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    weighted_ratings = pd.DataFrame(columns=['Weighted Rating'])
    for product in user_item_matrix.columns:
        product_ratings = user_item_matrix[product]
        weighted_rating = (similarities_df.dot(product_ratings) / similarities_df.sum(axis=1)).mean()
        weighted_ratings.loc[product] = [weighted_rating]

    top_products = weighted_ratings['Weighted Rating'].nlargest(5).index.tolist()

    products_info = []  # Initialize the list to hold product info dictionaries
    for product_name in top_products:
        product_data = df[df['Product Name'] == product_name].iloc[0]  # Fetching the first match for each product name
        product_url = product_data['Image URL']  # Extracting the URL from the product data
    
        # Append the product name and URL as a dictionary to the products_info list
        products_info.append({'product_name': product_name, 'product_url': product_url})

    return products_info

def extract_age(age_str):
    """
    Extracts numerical age from the given age string.
    
    Args:
        age_str (str): Age string in the format '(min-max)'.
    
    Returns:
        int: Numerical age if extraction is successful, None otherwise.
    """
    age_match = re.search(r'\d+', age_str)
    if age_match:
        return int(age_match.group())
    else:
        return None

def getFaceBox(net, frame, conf_threshold=0.5):
    """
    Detects faces in the given frame using the specified neural network.
    
    Args:
        net: Loaded neural network for face detection.
        frame: Image frame in which faces are to be detected.
        conf_threshold (float): Confidence threshold for face detection.
    
    Returns:
        tuple: Tuple containing processed frame and bounding boxes of detected faces.
    """
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def age_gender_detector(frame):
    """
    Detects age and gender of faces in the given frame.
    
    Args:
        frame: Image frame containing faces.
    
    Returns:
        tuple: Tuple containing processed frame, detected age, and detected gender.
    """
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        return frame, None, None  # If no faces are detected, return original frame and None for age/gender
    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age_str = ageList[agePreds[0].argmax()]
        age = extract_age(age_str)  # Extract numerical age from age string

        if age is not None:  # Check if age extraction was successful
            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            logging.warning(f"Failed to extract numerical age from string: {age_str}")
    return frameFace, age, gender

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint to upload an image for age and gender detection.
    
    Returns:
        Response: Processed image with age and gender labels.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        processed_frame, detected_age, detected_gender = age_gender_detector(frame)
        
        _, buffer = cv2.imencode('.jpg', processed_frame)
        io_buf = io.BytesIO(buffer)
        
        response = send_file(io_buf, mimetype='image/jpeg')
        response.headers['Age'] = str(detected_age) if detected_age is not None else 'Unknown'
        response.headers['Gender'] = detected_gender if detected_gender is not None else 'Unknown'
        return response
    except Exception as e:
        logging.error("Error processing image: ", exc_info=True)
        return jsonify({'error': 'Internal server error while processing image'}), 500

@app.route('/recommend', methods=['POST'])
def recommend_products():
    """
    Endpoint to recommend products based on user's age and gender.
    
    Returns:
        Response: JSON response containing recommended products.
    """
    data = request.json
    age = data.get('age')
    gender = data.get('gender')
    if not all([age, gender]):
        return jsonify({'error': 'Age and/or gender not provided'}), 400

    try:
        data_path = 'https://raw.githubusercontent.com/7Ritika/MM802_FinalProject/main/User_data.csv'

        recommendations = recommend_product_images_for_age_gender_with_fallback(age, gender, data_path)
        return jsonify({'age': age, 'gender': gender, 'recommendations': recommendations}), 200
    except Exception as e:
        logging.error("Error generating recommendations: ", exc_info=True)
        return jsonify({'error': 'Internal server error generating recommendations'}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
