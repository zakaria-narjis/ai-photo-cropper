import io
import time
from zipfile import ZipFile

import streamlit as st
from PIL import Image
from streamlit import session_state as ss
from streamlit_cropper import st_cropper

from api_handler import CompositionAPI as api
from user_session import User
from utils import draw_rectangle, image_to_byte_array

# Manual Cropping Parameters
BOX_COLOR = "#FF0000"
ASPECT_RATIO = None
STROKE_WIDTH = 2
REALTIME_UPDATE = True

# One crop API delay
API_DELAY = 1


if "user" not in ss:
    ss.user = User()
    ss.multi_crop = False
    ss.ai_crop = True
    ss.user.image_file = None

st.set_option("deprecation.showfileUploaderEncoding", False)

st.header("Composition Aware Cropping")

ss.ai_crop = st.toggle(label="AI Powered Cropping", value=True)

if ss.ai_crop:
    ss.multi_crop = st.toggle("Multi image cropping", disabled=not ss.ai_crop)

ss.user.image_file = st.file_uploader(
    label=["Upload an image", "Upload your images"][ss.multi_crop],
    type=["png", "jpg"],
    accept_multiple_files=ss.multi_crop,
)

if ss.multi_crop is False and ss.ai_crop is True:
    sample_image = st.selectbox(
        "Choose a sample image",
        ("University_Lubeck", "Frozen_lubeck_1", "Frozen_lubeck_2", "Lubeck_Night"),
    )
    if st.button("Try sample"):
        image_file = Image.open(f"sample_images/{sample_image.lower()}.jpg")
        ss.user.image_file = io.BytesIO((image_to_byte_array(image_file)))

        if ss.multi_crop:
            ss.user.image_file = [ss.user.image_file]

if ss.multi_crop:
    if ss.user.image_file != []:
        crop_download = st.button("Crop & Download")
        if crop_download:
            with st.spinner("Generating image crops with AI magic, please wait..."):
                crops = api.multi_crop(ss.user.image_file)

            progress_text = "Cropping images progress. Please wait."
            progress_bar = st.progress(0, text=progress_text)

            zip_file_bytes_io = io.BytesIO()
            with ZipFile(zip_file_bytes_io, "w") as zip_file:
                num_total_files = len(ss.user.image_file)
                for index, (image, crop_result) in enumerate(
                    zip(ss.user.image_file, crops["crops"])
                ):
                    img = Image.open(image)
                    cropped_img = img.crop(list(crop_result["coords"].values()))
                    cropped_img = image_to_byte_array(cropped_img)
                    zip_file.writestr(f"images/{crop_result['image_name']}.png", cropped_img)
                    progress = int(((index + 1) * 100 / num_total_files))
                    progress_bar.progress(progress, progress_text)

            progress_bar.empty()

            st.download_button(
                label="Download cropped images",
                data=zip_file_bytes_io,
                file_name="cropped_images.zip",
                mime="application/zip",
            )

else:
    if ss.user.image_file is not None:
        img = Image.open(ss.user.image_file)

        try:
            file_name = ss.user.image_file.name.split(".")[0]
            file_name = f"{file_name}.png"
        except AttributeError:
            file_name = sample_image

        if ss.ai_crop:
            with st.spinner("Cropping image with AI magic, please wait..."):
                ss.user.recommended_crop = api.one_crop(ss.user.image_file)
                time.sleep(API_DELAY)
            st.success("Done!")
            annotated_img = draw_rectangle(img, ss.user.recommended_crop)
            st.image(annotated_img, caption=file_name)
            cropped_img = img.crop(list(ss.user.recommended_crop.values()))
        else:
            cropped_img = st_cropper(
                img,
                realtime_update=REALTIME_UPDATE,
                box_color=BOX_COLOR,
                aspect_ratio=ASPECT_RATIO,
                stroke_width=STROKE_WIDTH,
                default_coords=None,
                return_type="image",
            )
            st.write("Preview")
            cropped_img.thumbnail((150, 150))
            st.image(cropped_img)

        st.download_button(
            label="Download image",
            data=image_to_byte_array(cropped_img),
            file_name=f"Cropped_{file_name}",
            mime="image/png",
        )
