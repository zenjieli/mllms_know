import os.path as osp
import sys
# Add this file's parent directory to the Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import requests



# device = f'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True, attn_implementation="eager").eval().to(device, torch.bfloat16)

model.eval()

image_path = './images/demo1.png'
question = 'what is the date of the photo?'

query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': f'{question} Answer:'},
])

general_query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': "Write a general description of the image. Answer:"},
])


with torch.no_grad():
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)

    pos = inputs['input_ids'].tolist()[0].index(151857) + 1

    general_inputs = tokenizer(general_query, return_tensors='pt')
    general_inputs = general_inputs.to(model.device)

    extract_attention_weights = None

    def attention_hook(module, input, output):
        global extract_attention_weights
        extract_attention_weights = output[1]

    handle = model.transformer.visual.attn_pool.attn.register_forward_hook(attention_hook)

    output = model(**inputs, output_attentions=True)

    att_resample = extract_attention_weights[0]

    general_output = model(**general_inputs, output_attentions=True)

    general_att_resample = extract_attention_weights[0]

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))

    final_atts = []

    for i, ax in enumerate(axes.flatten()):

        att = output.attentions[i][0,:,-1,pos:pos+256].mean(dim=0)
        att = att @ att_resample
        att = att.to(torch.float32).detach().cpu().numpy()

        general_att = general_output.attentions[i][0,:,-1,pos:pos+256].mean(dim=0)
        general_att = general_att @ general_att_resample
        general_att = general_att.to(torch.float32).detach().cpu().numpy()

        att = att / general_att

        final_atts.append(att)
        ax.imshow(att.reshape(32,32), cmap='viridis', interpolation='nearest')
        ax.set_title(f'Layer {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

img = Image.open(image_path)
img.save('./output/qwen_demo1.png')