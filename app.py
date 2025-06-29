import streamlit as st
import numpy as np
#import cv2
from PIL import Image
import mediapipe as mp
import os
# st.set_page_config(page_title="Auto-align Hair & Beard", layout="centered")
# st.title("🧔‍♂️ Auto-align Hair & Beard with MediaPipe")
#
# # Load user images
# face_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
# hair_file = st.file_uploader("Upload Hair Style Image", type=["jpg","jpeg","png"])#, help="Use transparent PNG")
# beard_file = st.file_uploader("Upload Beard Style Image", type=["jpg","jpeg","png"])#, help="Use transparent PNG")

def apply_beard_overlay(beard_overlay_path,capturephoto_path):
    overlay_image = beard_overlay_path#Image.open(beard_overlay_path)  # RGBA
    overlay_image1=overlay_image
    overlay_image=np.array(overlay_image)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    # Drawing utilities (optional)
    # mp_drawing = mp.solutions.drawing_utils
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    frame = capturephoto_path#Image.open(capturephoto_path)
    frame1=frame
    w,h = frame.size
    frame = np.array(frame)
    rgb_frame = frame1.convert("RGB")
    rgb_frame = np.array(rgb_frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for eyes (example using landmark 33 and 263 for left and right eye corners)
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            # Use jaw (landmarks 152, 234, 454) to estimate position and scale
            chin = landmarks[2]
            left = landmarks[177]
            right = landmarks[401]

            beard_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.35)
            beard_height = int((beard_width * overlay_image.shape[0] / overlay_image.shape[1]) * 1)

            x = chin[0] - beard_width // 2
            y = chin[1] - beard_height // 3  # Move up a bit
            resized_overlay = overlay_image1.resize((beard_width, beard_height))#, Image.ANTIALIAS)
            resized_overlay1=resized_overlay
            resized_overlay = np.array(resized_overlay)
            # Ensure coordinates are in frame bounds
            x_start = max(0, x)
            y_start = max(0, y)
            for i in range(0, resized_overlay.shape[0], 1):
                for j in range(0, resized_overlay.shape[1], 1):
                    if resized_overlay[i][j][0] > 175 and resized_overlay[i][j][1] > 175 and resized_overlay[i][j][
                        2] > 175:
                        resized_overlay[i][j][0] = 255
                        resized_overlay[i][j][1] = 255  # white_img[i][j][1]
                        resized_overlay[i][j][2] = 255
            r, g, b, a = resized_overlay1.split()
            #resized_overlay = cv2.merge((b, g, r))
            resized_overlay = Image.merge("RGB", (r, g, b))
            resized_overlay = np.array(resized_overlay)
            for i in range(0, resized_overlay.shape[0], 1):
                for j in range(0, resized_overlay.shape[1], 1):
                    if resized_overlay[i][j][0] == 255 and resized_overlay[i][j][1] == 255 and \
                            resized_overlay[i][j][2] == 255:
                        resized_overlay[i][j][0] = frame[y_start + i][x_start + j][0]
                        resized_overlay[i][j][1] = frame[y_start + i][x_start + j][1]  # white_img[i][j][1]
                        resized_overlay[i][j][2] = frame[y_start + i][x_start + j][2]  # white_img[i][j][2]
                    # frame[y_start + i, x_start + j, c]=resized_overlay[i, j, c]
            frame[y_start:y_start + resized_overlay.shape[0],
            x_start:x_start + resized_overlay.shape[1]] = resized_overlay
    frame = Image.fromarray(frame)
    return frame
    # frame.save(output_path)
    # return output_path

def apply_hair_overlay(hair_overlay_path,capturephoto_path):
    overlay_image = hair_overlay_path #Image.open(hair_overlay_path)  # RGBA
    overlay_image1=overlay_image
    overlay_image = np.array(overlay_image)
    # Mediapipe face mesh setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Drawing utilities (optional)
    # mp_drawing = mp.solutions.drawing_utils
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # OpenCV video capture
    frame = capturephoto_path  #Image.open(capturephoto_path)
    frame1=frame
    frame = np.array(frame)
    h, w, _ = frame.shape
    rgb_frame = frame1.convert("RGB")
    rgb_frame = np.array(rgb_frame)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for eyes (example using landmark 33 and 263 for left and right eye corners)
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

            forehead = landmarks[151]  # 10 # Top of forehead
            left_side = landmarks[234]  # Left side of face
            right_side = landmarks[454]  # Right side of face

            hair_width = int(np.linalg.norm(np.array(left_side) - np.array(right_side)) * 1.7)
            hair_height = int(hair_width * overlay_image.shape[0] / overlay_image.shape[1])

            x = forehead[0] - hair_width // 2
            y = forehead[1] - hair_height  # Move above forehead

            # Ensure coordinates are in bounds
            x_start = max(0, x)
            y_start = max(0, y)
            #resized_overlay = cv2.resize(overlay_image, (hair_width, hair_height))
            resized_overlay = overlay_image1.resize((hair_width, hair_height))  # , Image.ANTIALIAS)
            resized_overlay1 = resized_overlay
            resized_overlay = np.array(resized_overlay)
            # Ensure coordinates are in frame bounds

            for i in range(0, resized_overlay.shape[0], 1):
                for j in range(0, resized_overlay.shape[1], 1):
                    if resized_overlay[i][j][0] > 175 and resized_overlay[i][j][1] > 175 and resized_overlay[i][j][
                        2] > 175:
                        resized_overlay[i][j][0] = 255
                        resized_overlay[i][j][1] = 255  # white_img[i][j][1]
                        resized_overlay[i][j][2] = 255
            # resized_hairstyle = cv2.resize(hairstyle, (150, 150))
            r, g, b, a = resized_overlay1.split()
            # resized_overlay = cv2.merge((b, g, r))
            resized_overlay = Image.merge("RGB", (r, g, b))
            resized_overlay = np.array(resized_overlay)
            # alpha = a / 255.0
            for i in range(0, resized_overlay.shape[0], 1):
                for j in range(0, resized_overlay.shape[1], 1):
                    if resized_overlay[i][j][0] == 255 and resized_overlay[i][j][1] == 255 and \
                            resized_overlay[i][j][
                                2] == 255:
                        resized_overlay[i][j][0] = frame[y_start + i][x_start + j][0]
                        resized_overlay[i][j][1] = frame[y_start + i][x_start + j][1]  # white_img[i][j][1]
                        resized_overlay[i][j][2] = frame[y_start + i][x_start + j][2]  # white_img[i][j][2]
                    # frame[y_start + i, x_start + j, c]=resized_overlay[i, j, c]
            frame[y_start:y_start + resized_overlay.shape[0],
            x_start:x_start + resized_overlay.shape[1]] = resized_overlay
    frame = Image.fromarray(frame)
    return frame
    # frame.save(output_path)
    # return output_path

st.set_page_config(page_title="Auto-align Hair & Beard", layout="centered")
st.title("🧔‍♂️ Auto-align Hair & Beard with MediaPipe")

HAIR_STYLE_DIR = "./static"
BEARD_STYLE_DIR = "./static"

# Load all hairstyle and beard images
hair_styles = [f for f in os.listdir(HAIR_STYLE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
beard_styles = [f for f in os.listdir(BEARD_STYLE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]


# # Load user images
# face_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
# hair_file = st.file_uploader("Upload Hair Style Image", type=["jpg","jpeg","png"])#, help="Use transparent PNG")
# beard_file = st.file_uploader("Upload Beard Style Image", type=["jpg","jpeg","png"])#, help="Use transparent PNG")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎨 Select Hair Style")
    hair_choice = st.radio("Choose Hair Style", hair_styles, format_func=lambda x: x.split('.')[0])
    hair_file = os.path.join(HAIR_STYLE_DIR, hair_choice)
    st.image(hair_file, width=150)

with col2:
    st.subheader("🧔 Select Beard Style")
    beard_choice = st.radio("Choose Beard Style", beard_styles, format_func=lambda x: x.split('.')[0])
    beard_file = os.path.join(BEARD_STYLE_DIR, beard_choice)
    st.image(beard_file, width=150)

st.markdown("---")
face_file = st.file_uploader("📸 Upload Face Image", type=["jpg", "jpeg", "png"])

if face_file is not None:
    base_img = Image.open(face_file)

    if hair_file is not None:
        hair_img = Image.open(hair_file)
        base_img = apply_hair_overlay(hair_img, base_img)

    if beard_file is not None:
        beard_img = Image.open(beard_file)
        base_img = apply_beard_overlay(beard_img, base_img)

    st.subheader("🔍 Auto-aligned Result")
    st.image(base_img, use_column_width=True)
else:
    st.warning("👤 Please upload a face image to begin.")
