import streamlit as st
from user_session import User
from streamlit import session_state as ss
from PIL import Image
from api_hundler import ClipCropAPI as api
from utils import image_to_byte_array, draw_rectangle


if 'user' not in ss:
  ss.user = User()
  ss.multi_crop = False
  ss.ai_crop = True
  ss.user.image_file = None
  ss.user.query = None

st.header("Clip Crop : Extract sections of your images")

with st.form(key='clip crop from'):
    ss.user.image_file = st.file_uploader(
                                label='Upload an image', 
                                type=['png', 'jpg'], 
                                accept_multiple_files=False
                                )

    ss.user.query = st.text_input(label = 'Enter Keyword for Image Crop',
                                help="Type a keyword that describes the object or section you want to crop from the image.",
                                placeholder="e.g., 'tree', 'logo'")
    submited = st.form_submit_button(label='Crop')

if submited:
    if ss.user.image_file!= None and ss.user.query!=None :
        img = Image.open(ss.user.image_file)
        with st.spinner('Cropping image with AI magic, please wait...'):
            #get the crop from the back end
            ss.user.recommended_crop  = api.crop(ss.user.image_file,ss.user.query)
        st.success('Done!')
        annotated_img = draw_rectangle(img,ss.user.recommended_crop)
        st.image(annotated_img, caption=f'{ss.user.image_file.name}.png')
        cropped_img = img.crop(list(ss.user.recommended_crop.values()))
        st.download_button(
                    label="Download image",
                    data=image_to_byte_array(cropped_img),
                    file_name=f"Cropped_{ss.user.image_file.name}.png",
                    mime="image/png"
                )