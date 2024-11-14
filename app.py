import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array


import sys
sys.path.append('./efficientnet_keras_transfer_learning')

# model
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model


get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model = load_model('../Multi_app/data/16_Multi_8e-6_250_Unfreeze.h5') #paper
#-------------------------------------------------------------------
# cut_image.py
def cut_image(image):
    width, height = image.size
    frac=0.6
    crop_left_width = int(width * frac)
    cropped_left = image.crop((0, 0, crop_left_width, height))
    crop_right_width = width - crop_left_width
    cropped_right = image.crop((crop_right_width, 0, width, height)).transpose(Image.FLIP_LEFT_RIGHT)
    return cropped_left, cropped_right
#---------------------------------------------------------------------------------
# ฟังก์ชันการทำนาย
#def predict_image(img_path,model, height, width):
    # Read the image and resize it
    #img = image.load_img(img_path, target_size=(height, width))
    # Convert it to a Numpy array with target shape.
    #x = image.img_to_array(img)
    # Reshape
    #x = x.reshape((1,) + x.shape)
    #x /= 255.
    #result = model.predict([x])
    #return result

def predict_image(img, model, height, width):
    # ปรับขนาดและแปลงเป็นอาร์เรย์
    img = img.resize((height, width))  # ปรับขนาดภาพ
    x = img_to_array(img) / 255.0  # Normalize ภาพ
    x = np.expand_dims(x, axis=0)  # เพิ่มมิติ batch
    result = model.predict(x)
    return result
#----------------------------------------------------------------------------------
#หาค่า confident
def calculate_confident(value):
    if value >= 0.5: #male
        confident = value
    else:
        confident = 1 - value #female
    return confident

#----------------------------------------------------------------------------------
# ฟังก์ชันทำนายอายุและเพศ
def predict_age_gender(img_paths):
    pred_list_regression = []
    pred_list_classification = []

    for img in img_paths:
        height = width = model.input_shape[1]
        predictions = predict_image(img, model, height, width)
        regression_result = predictions[0]
        classification_result = predictions[1]
        pred_list_regression.append(regression_result)
        pred_list_classification.append(classification_result)

    # คำนวณความมั่นใจ
    con_0 = calculate_confident(pred_list_classification[0][0][0])
    con_1 = calculate_confident(pred_list_classification[1][0][0])
    
    # เลือกภาพที่มีค่าความมั่นใจสูงสุด
    if con_0 > con_1:
        gender_pred = "Male" if pred_list_classification[0][0][0] >= 0.5 else "Female"
        age_pred = np.around(pred_list_regression[0][0][0])
    else:
        gender_pred = "Male" if pred_list_classification[1][0][0] >= 0.5 else "Female"
        age_pred = np.around(pred_list_regression[1][0][0])

    confidence = max(con_0, con_1)
    return age_pred, gender_pred, confidence

#--------------------------------------------------------------------------------------
# เก็บรูป
import os
if not os.path.exists("uploads"):
    os.makedirs("uploads")
#-------------------------------------------------------------------------------------

# UI
st.markdown("<h1 style='text-align: center;'> 🦷 Age and Sex Estimation via Panoramic X-ray Image</h1>", unsafe_allow_html=True)
st.write("") 
st.write("") 

# รับภาพจากผู้ใช้
uploaded_file = st.file_uploader("Choose a dental X-ray image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    filename = uploaded_file.name  # Get the filename
    st.image(image, caption=f"Uploaded X-ray image: {filename}", use_column_width=True)

   # บันทึกภาพลงในโฟลเดอร์ "uploads"
    image_path = os.path.join("uploads", filename)
    image.save(image_path)

    # ตัดภาพเป็นภาพซ้ายและขวา
    left_img, right_img = cut_image(image)
    
    # เรียกฟังก์ชันทำนายอายุและเพศจากภาพทั้งสอง
    age, gender, confidence = predict_age_gender([left_img, right_img])

    # แสดงผลการทำนาย
    st.subheader("Prediction Results")
    st.write(f"<span style='font-size:24px;'> <b>Age</b>: <span style='color:green;'><b>{int(age)}</b></span> </span><span style='font-size:20px;'>years </span>", unsafe_allow_html=True)
    st.write(f"<span style='font-size:24px;'> <b>Sex</b>: <span style='color:green;'><b>{gender}</b></span> </span><span style='font-size:20px;'>(Confidence: <span style='color:blue;'>{confidence*100:.2f}%)</span> </span>", unsafe_allow_html=True)
