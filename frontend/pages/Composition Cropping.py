import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import io
from api_hundler import CompositionAPI as api

def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format='png')
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

st.set_option('deprecation.showfileUploaderEncoding', False)

st.header("Composition aware Cropping")
multi_on = st.toggle("Multi image cropping")
img_file = st.file_uploader(label=['Upload an image','Upload your images'][multi_on], type=['png', 'jpg'], accept_multiple_files=multi_on)


box_color = '#FF0000'
aspect_ratio = None
stroke_width = 2
realtime_update = True

if multi_on:
   pass
else:
  if img_file:
      img = Image.open(img_file)
      coords = api.one_crop(img_file)   
      # Get a cropped image from the frontend
      recommended_coords = [coords['x1'],coords['x2'],coords['y1'],coords['y2']]
      print(list(coords.values()))
  
      cropped_img = st_cropper(img, 
                               realtime_update=realtime_update, 
                               box_color=box_color,
                               aspect_ratio=aspect_ratio,
                               stroke_width=stroke_width,
                               default_coords = recommended_coords,
                               return_type='box',
                               )    
      print(cropped_img)
      st.download_button(
              label="Download image",
              data=image_to_byte_array(cropped_img),
              file_name=f"Cropped_{img_file.name}.png",
              mime="image/png"
            )
      st.button(
              label = "Reset to AI recommended crop")
      # Manipulate cropped image at will
      st.write("Preview")
      cropped_img.thumbnail((150,150))
      st.image(cropped_img)
