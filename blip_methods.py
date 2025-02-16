import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce
from utils import *

# hyperparameters
NUM_IMG_TOKENS = 256
NUM_PATCHES = 16
PATCH_SIZE = 14
IMAGE_RESOLUTION = 224
QFORMER_IMG_TOKENS = 32
QFORMER_LAYER = 2
LM_LAYER = 15

def gradient_attention_blip(image, prompt, general_prompt, model, processor):

    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    outputs = model(**inputs, output_attentions=True)

    # Compute logits and loss
    CE = nn.CrossEntropyLoss()
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -CE(zero_logit, true_class)


    # Extract attention maps
    q_former_atts = outputs.qformer_outputs.cross_attentions
    lm_atts = outputs.language_model_outputs.attentions

    # Compute gradients
    q_former_grads = torch.autograd.grad(loss, q_former_atts, retain_graph=True)
    lm_att_grads = torch.autograd.grad(loss, lm_atts, retain_graph=True)

    # Process Query Former attention maps
    q_former_att = q_former_atts[QFORMER_LAYER][0, :, :, 1:]
    q_former_grad = q_former_grads[QFORMER_LAYER][0, :, :, 1:]
    q_former_grad_att = (q_former_att * F.relu(q_former_grad)).mean(dim=0).unsqueeze(0)


    # Process Language Model attention maps
    lm_att = lm_atts[LM_LAYER][0, :, -1, :NUM_IMG_TOKENS]
    lm_grad = lm_att_grads[LM_LAYER][0, :, -1, :NUM_IMG_TOKENS]
    lm_grad_att = (lm_att * F.relu(lm_grad)).mean(dim=0).unsqueeze(0).unsqueeze(0)


    # Compute combined attention map
    att_map = torch.bmm(lm_grad_att, q_former_grad_att).squeeze(1).to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    return att_map

def rel_attention_blip(image, prompt, general_prompt, model, processor):
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(images=image, text=general_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

    outputs = model(**inputs, output_attentions=True)
    general_outputs = model(**general_inputs, output_attentions=True)

    # Extract attention maps
    q_former_atts = outputs.qformer_outputs.cross_attentions
    lm_atts = outputs.language_model_outputs.attentions

    general_q_former_atts = general_outputs.qformer_outputs.cross_attentions
    general_lm_atts = general_outputs.language_model_outputs.attentions

    # Process Query Former attention maps (4th layer)
    q_former_att = q_former_atts[QFORMER_LAYER][0, :, :, 1:].mean(dim=0).unsqueeze(0)
    general_q_former_att = general_q_former_atts[QFORMER_LAYER][0, :, :, 1:].mean(dim=0).unsqueeze(0)

    # Process Language Model attention maps (15th layer)
    lm_atts = lm_atts[LM_LAYER][0, :, -1, :NUM_IMG_TOKENS].mean(dim=0).unsqueeze(0).unsqueeze(0)
    general_lm_atts = general_lm_atts[LM_LAYER][0, :, -1, :NUM_IMG_TOKENS].mean(dim=0).unsqueeze(0).unsqueeze(0)

    # Compute combined attention maps
    att = torch.bmm(lm_atts, q_former_att).squeeze(1)
    general_att = torch.bmm(general_lm_atts, general_q_former_att).squeeze(1)

    # Compute attention map ratio
    att_map = att / general_att

    # Convert attention map to numpy and reshape
    att_map = att_map.to(torch.float32).detach().cpu().numpy().reshape(NUM_PATCHES, NUM_PATCHES)

    return att_map

def pure_gradient_blip(image, prompt, general_prompt, model, processor):
    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    general_inputs = processor(images=image, text=general_prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    
    # Apply high pass filter
    high_pass = high_pass_filter(image, IMAGE_RESOLUTION, reduce=False)
    
    # Enable gradients
    inputs['pixel_values'].requires_grad = True
    general_inputs['pixel_values'].requires_grad = True
    
    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass for inputs
    outputs = model(**inputs, output_hidden_states=False)
    zero_logit = outputs.logits[:, -1, :]
    true_class = torch.argmax(zero_logit, dim=1)
    loss = -criterion(zero_logit, true_class)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, inputs['pixel_values'], retain_graph=True)[0]
    
    # Forward pass for general_inputs
    general_outputs = model(**general_inputs, output_hidden_states=False)
    general_zero_logit = general_outputs.logits[:, -1, :]
    general_true_class = torch.argmax(general_zero_logit, dim=1)
    general_loss = -criterion(general_zero_logit, general_true_class)
    
    # Compute general gradients
    general_grads = torch.autograd.grad(general_loss, general_inputs['pixel_values'], retain_graph=True)[0]
    
    # Process gradients
    grads = grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    general_grads = general_grads.to(torch.float32).detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Compute gradient norms
    grad = np.linalg.norm(grads, axis=2)
    general_grad = np.linalg.norm(general_grads, axis=2)
    
    # Normalize and apply high pass filter
    grad = grad / general_grad
    high_pass = high_pass > np.median(high_pass)
    grad = grad * high_pass
    
    # Reduce gradient block size
    grad = block_reduce(grad, block_size=(PATCH_SIZE, PATCH_SIZE), func=np.mean)
    
    return grad