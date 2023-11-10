# -*- coding:utf-8 -*-

import argparse
import os
import re
import sys
import uuid

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from addict import Dict
from one_model.common.config import Config
from one_model.inference.infer import Infer

# Gradio
output_labels = ["Segmentation Output"]

title = "One Model: Data Mining via Large Language Model"

description = """
<font size=4>
This is the online demo of One Model. \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Note**: **Different prompts can lead to significantly varied results**. \n
**Note**: Please try to **standardize** your input text prompts to **avoid ambiguity**, and also pay attention to whether the **punctuations** of the input are correct. \n
**Note**: Current model is **One Model-13B-llama2-v0-explanatory**, and 4-bit quantization may impair text-generation quality. \n
**Usage**: <br>
&ensp;(1) To let One Model **segment something**, input prompt like: "Segment the xxx"; <br>
&ensp;(2) To let One Model **segment everything**, input prompt like: "Can you segment everything in this image?"; <br>
&ensp;(3) To let One Model **output a description**, input prompt like: "describe visible details about these objects and describe the interrelationships among these objects."; <br>
&ensp;(4) To let One Model **output a generated image**, input prompt like: "Generate a image of an empty flatbed"; <br>
Hope you can enjoy our work! <br>
The model has not been retrained yet, but the engineering architecture has met the requirements. The input prompt is based on the similarity of CLIP to determine what type of task to use. Later, through training, LLM can determine what tasks to use downstream.
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://git.xiaojukeji.com/qiudanwang_i/one-model' target='_blank'> Gitlab Repo </a></p>
"""

# Create model
cfg_path = "configs/infer/config_13B_decoders.yaml"
config: Config = Config(Dict(cfg_path=cfg_path))
infer = Infer(config)
infer.init_model()

input_save_dir = "./vis_input"
output_save_dir = "./vis_output"

## to be implemented
def inference(input_str, input_image):
    # filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    # input valid check
    if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
        output_str = "[Error] Invalid input: ", input_str
        output_image = cv2.imread("./resources/error_happened.png")[:, :, ::-1]
        return output_image, output_str
    # save input image
    image_id = str(uuid.uuid1())
    print("image_id: ", image_id)
    input_image = cv2.imread(input_image)
    input_image_path = input_save_dir + "/" + image_id + ".png"
    if input_image is not None:
        cv2.imwrite(input_image_path, input_image)
    else:
        input_image = cv2.imread("./resources/no_seg_out.png")
        cv2.imwrite(input_image_path, input_image)
    # Model Inference
    output_str, output_image_path = infer.predict(input_image_path, input_str, output_save_dir)
    output_image = cv2.imread(output_image_path)
    if output_image is None:
       output_image = cv2.imread("./resources/no_seg_out.png") 
    output_image = output_image[:, :, ::-1]
    return output_str, output_image


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Output"),
        gr.Image(type="pil", label="Segmentation Output"),
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging="auto",
)

demo.queue()
demo.launch(share=False, server_name="0.0.0.0", server_port=8085, ssl_verify=False)
