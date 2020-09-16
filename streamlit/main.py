import os
import shutil
import subprocess

import SessionState
import cv2
import streamlit as st
from PIL import Image

st.title("Thanh_Passport")

# image_upload=st.file_uploader(label="Insert Image here:",type="jpg")
path = st.text_input("Or Insert Path:")
# if path ==None and image_upload==None:
#     st.warning("Invalid input")
#     st.stop()
#     st.success("Nice input")
#
# if image_upload !=None:
#     path=None
#     image=Image.open(image_upload)
#     image.save("object_detection/passport.jpg")
#     image_upload=None

if path!=None:
    image_upload=None
    session=SessionState.get(count=0)
    try:
        image_names=os.listdir(path)
        image_names = [image_name for image_name in image_names if image_name[-3:] == "jpg"]
        image = Image.open(os.path.join(path, image_names[session.count]))
        image.save("detect_passport/passport.jpg")
        if st.button("Next"):
            session.count += 1
            session.count = session.count % len(image_names)

        if st.button("Back"):
            session.count -= 1
            session.count = session.count % len(image_names)
        image = Image.open(os.path.join(path, image_names[session.count]))
    except:
        st.text("Invalid path")
    path = None

# show image
try:
    image.save("detect_passport/passport.jpg")
    st.image(Image.open("detect_passport/passport.jpg"))
except:
    st.text("Invalid image")

# detect passport
if st.button("Detect passport"):
    os.chdir('detect_passport')
    subprocess.call(["python3", "detect_passport.py"])
    st.write("Detect complete. Passport in detect_passport/passport.jpg")
    st.image(Image.open("result.jpg"))
    os.chdir("../")

# detect and predict
if st.button("Detect info"):
    # detect code here
    os.chdir("object_detection")
    if os.path.exists("object"):
        shutil.rmtree("object")
    os.mkdir("object")
    subprocess.call(["python3", "detect.py"])
    st.write("Detect complete. Images in attention_ocr/data")
    st.image(Image.open("result.jpg"), "Detection result: ")
    os.chdir("../")

if st.button("OCR"):
    #ocr code here
    os.chdir("attention_ocr/")
    subprocess.call(["python3","ocr.py"])
    st.write("OCR complete. Results in attention/ocr_result.txt")
    os.chdir("../")

    predicts=[]
    for line in open("attention_ocr/ocr_result.txt"): predicts.append(line[:-1])
    for i,image_name in enumerate(sorted(os.listdir("object_detection/object"))):
        class_text=image_name.split("-")[-1][:-4]
        image=cv2.imread("object_detection/object/"+image_name)
        h,w=image.shape[:2]
        newh = 50
        neww = int(newh / h * w)
        image = cv2.resize(image, (neww, newh))
        st.image(image,class_text+": "+predicts[i])

