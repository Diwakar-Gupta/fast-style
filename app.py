import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import streamlit as st
from tempfile import NamedTemporaryFile

st.write('https://github.com/Diwakar-Gupta/fast-style')

def crop_center(image):
  """Return a croped squared image"""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1]-shape[2], 0) // 2
  offset_x = max(shape[2]-shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape
  )
  return image

def load_image(img_path, image_size=(256,256), preserve_aspect_ration=True):
  """Loading and preprocessing Images."""
  # img_path = tf.keras.utils.get_file(origin=image_url)
  img = tf.io.read_file(img_path)
  img = tf.io.decode_image(img, channels=3, dtype=tf.float32)[tf.newaxis, ...]

  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

output_image_size = 384  

@st.cache
def get_hub():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
    return hub_module

# The content image size can be any think.
content_img_size = (output_image_size, output_image_size)

getter1, getter2 = st.columns(2)

content_image = getter1.file_uploader('Content Image in jpeg format')
style_image = getter2.file_uploader('Style Image in jpeg format')
style_img_size = (256, 256)

content, style, mixed = st.columns(3)

if content_image is not None:
    image = Image.open(content_image)
    content.image(image)

if style_image is not None:
    image = Image.open(style_image)
    style.image(image)

if content_image != None and style_image != None and  st.button('Generate Style'):
    temp_file = NamedTemporaryFile(delete=True)
    temp_file.write(content_image.getvalue())
    content_image = load_image(temp_file.name)

    temp_file = NamedTemporaryFile(delete=True)
    temp_file.write(style_image.getvalue())
    style_image = load_image(temp_file.name)

    outputs = get_hub()(content_image, style_image)
    sytlized_image = outputs[0]

    mixed.image(sytlized_image.numpy())
else:
    st.write('Result Image')
