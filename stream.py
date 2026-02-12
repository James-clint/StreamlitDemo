import streamlit as st
import pandas as pd
import json, time, os, sys, time, requests
import torch, uuid
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image



model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def text_to_image(text):
  image = pipe(text).images[0]

  image_path = uuid.uuid4() + '.png'
  image.save(image_path)

  return image_path

user_input = st.text_input("Enter your text prompt")

if st.button("process"):
  if user_input:
    image_path = text_to_image(user_input)
    st.success("Message sent! waiting for response....")

    
    if image_path:
      st.image(image_path, caption='Generated Image', use_column_width=True)
       
    else:
      st.error("Failed to process image.")
  else:
    st.warning("Please enter a message before sending.")
