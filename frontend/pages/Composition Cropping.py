import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw 
import io
from api_hundler import CompositionAPI as api
from streamlit import session_state as ss
from user_session import User
import time
import numpy as np
from zipfile import ZipFile
from utils import image_to_byte_array, draw_rectangle

#Manual Cropping Parameters
box_color = '#FF0000'
aspect_ratio = None
stroke_width = 2
realtime_update = True

#One crop api delay
API_DELAY = 1


if 'user' not in ss:
  ss.user = User()
  ss.multi_crop = False
  ss.ai_crop = True
  ss.user.image_file = None

st.set_option('deprecation.showfileUploaderEncoding', False)

st.header("Composition Aware Cropping")


ss.ai_crop = st.toggle(label = "AI Powered Cropping", value=True)

if ss.ai_crop:
  ss.multi_crop = st.toggle("Multi image cropping",disabled=not ss.ai_crop)

ss.user.image_file = st.file_uploader(
                            label=['Upload an image','Upload your images'][ss.multi_crop], 
                            type=['png', 'jpg'], 
                            accept_multiple_files=ss.multi_crop
                            )


if ss.multi_crop :
  if ss.user.image_file != [] :  
    crop_download = st.button("Crop & Download")
    if crop_download: 
      with st.spinner('Generating image crops with AI magic, please wait...'):
        crops = api.multi_crop(ss.user.image_file)

      progress_text = "Cropping images progress. Please wait."
      progress_bar = st.progress(0, text=progress_text)

      cropped_imgs = []
      zip_file_bytes_io = io.BytesIO()
      with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        num_total_files = len(ss.user.image_file)
        for index,(image,crop_result) in enumerate(zip (ss.user.image_file,crops['crops'])):
            img = Image.open(image)
            cropped_img = img.crop(list(crop_result['coords'].values()))  
            cropped_img = image_to_byte_array(cropped_img)
            cropped_imgs.append(cropped_img)
            zip_file.writestr(f"images/{crop_result['image_name']}", cropped_img)
            progress = int(((index+1)*100/num_total_files) )
            progress_bar.progress(progress,progress_text )

      progress_bar.empty()
      
      st.download_button(
                label="Download cropped images",
                data=zip_file_bytes_io,
                file_name=f"cropped_images.zip",
                mime="application/zip"
              )   
    
else:
  if ss.user.image_file!= None :

    img = Image.open(ss.user.image_file)

    if ss.ai_crop:

      with st.spinner('Cropping image with AI magic, please wait...'):
        #get the crop from the back end
        ss.user.recommended_crop  = api.one_crop(ss.user.image_file )
        time.sleep(API_DELAY)
      st.success('Done!')
      annotated_img = draw_rectangle(img,ss.user.recommended_crop)
      st.image(annotated_img, caption=f'{ss.user.image_file.name}.png')
      cropped_img = img.crop(list(ss.user.recommended_crop.values()))
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

    st.download_button(
                label="Download image",
                data=image_to_byte_array(cropped_img),
                file_name=f"Cropped_{ss.user.image_file.name}.png",
                mime="image/png"
              )