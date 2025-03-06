import os.path as osp
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    'llava-hf/llava-1.5-7b-hf', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").to('cuda')
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

from PIL import Image, ImageDraw
import sys
# Add this file's parent directory to the Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from run import vicrop_qa

model_name = 'llava'
method_name = 'rel_att'
image_path = 'images/demo1.png'
question = 'what is the date of the photo?'
short_question = 'what is the date of the photo?'

# Run the Vicrop method
ori_answer, crop_answer, bbox = vicrop_qa(model_name, method_name, image_path, question, model, processor, short_question)

print(f'Model\'s original answer:  {ori_answer}')
print(f'Answer with Vicrop:       {crop_answer}')

# Visualize the bounding box
image = Image.open(image_path).convert("RGB")
image_draw = ImageDraw.Draw(image)
image_draw.rectangle(bbox, outline='red', width=4)
resized_img = image.resize((500, 500 * image.size[1] // image.size[0]))
# save resized image to file
resized_img.save('output/demo1_resized.png')
