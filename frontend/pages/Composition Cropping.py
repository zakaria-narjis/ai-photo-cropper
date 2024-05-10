import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw 
import io
from api_hundler import CompositionAPI as api
from streamlit import session_state as ss
from user_session import User
import time
import numpy as np


box_color = '#FF0000'
aspect_ratio = None
stroke_width = 2
realtime_update = True
API_DELAY = 1
if 'user' not in ss:
  print('taco')
  ss.user = User()
  ss.multi_crop = False
  ss.ai_crop = True
  ss.ai_crop_label = ["Manual Cropping","AI Powered Cropping"][ss.ai_crop]
def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format='png')
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

# def change_name():
#     ss.ai_crop_label = ["Manual Cropping","AI Powered Cropping"][ss.ai_crop]

def draw_rectangle(image: Image,bbox:dict)->Image:
  img = image.copy()
  draw = ImageDraw.Draw(img)
  draw.rectangle(((bbox["x1"], bbox["y1"]),(bbox["x2"], bbox["y2"])), outline='Red')
  return img

st.set_option('deprecation.showfileUploaderEncoding', False)

st.header("Composition Aware Cropping")
# toggle_label = ["Manual Cropping","AI Powered Cropping"][ss.ai_crop]

toggle_label = (
    "Manual Cropping"
    if st.session_state.get("ai_toggle", False)
    else "AI Powered Cropping"
)
toggle_value = st.session_state.get("ai_toggle", True)
ss.ai_crop = st.toggle(label = toggle_label, value=toggle_value, key="ai_toggle")
ss.multi_crop = st.toggle("Multi image cropping",disabled=not ss.ai_crop)

ss.user.image_file = st.file_uploader(
                            label=['Upload an image','Upload your images'][ss.multi_crop], 
                            type=['png', 'jpg'], 
                            accept_multiple_files=ss.multi_crop
                            )



if ss.user.image_file!= None :

  img = Image.open(ss.user.image_file)

  if ss.ai_crop:

    with st.spinner('Cropping image with AI magic, please wait...'):
      #get the crop from the back end
      ss.user.recommended_crop  = api.one_crop(ss.user.image_file )
      time.sleep(API_DELAY)
    st.success('Done!')

    # recommended_coords = [ss.user.recommended_crop['x1'], ss.user.recommended_crop['x2'], ss.user.recommended_crop['y1'], ss.user.recommended_crop['y2']]
    
    # cropped_img = st_cropper(img, 
    #                           realtime_update=realtime_update, 
    #                           box_color=box_color,
    #                           aspect_ratio=aspect_ratio,
    #                           stroke_width=stroke_width,
    #                           default_coords = recommended_coords,
    #                           return_type='image',
    #                           )    
    
    cropped_img = draw_rectangle(img,ss.user.recommended_crop)
    st.image(np.array(cropped_img), caption=f'{ss.user.image_file.name}.png')
    st.download_button(
            label="Download image",
            data=image_to_byte_array(cropped_img),
            file_name=f"Cropped_{ss.user.image_file.name}.png",
            mime="image/png"
          )

  else:
    cropped_img = st_cropper(img, 
                                realtime_update=realtime_update, 
                                box_color=box_color,
                                aspect_ratio=aspect_ratio,
                                stroke_width=stroke_width,
                                default_coords = None,
                                return_type='image',
                                )   
    st.write("Preview")
    cropped_img.thumbnail((150,150))
    st.image(cropped_img)
