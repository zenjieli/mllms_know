import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter
from skimage.measure import block_reduce


def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    # att_map: a numpy array with size of 24*24 (LLaVA) or 16*16 (BLIP)
    # image_size: the size of the original image
    # bbox_size: the size of the crop image

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, how many blocks do I need? (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)
        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att


def process_data(data_name, data_list, image_path):
    
    # load the raw json data then process into a list of dictionaries
    # [
    #    { 'image_path': image_path, 'question': 'question string', },
    #    ...    
    # ]
    data = []
    if data_name == 'textvqa':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, d['image_id'] + '.jpg'),
                'question': d['question'],
                'label': d['answers'],
            })
    elif data_name == 'vqav2':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, f'COCO_val2014_{str(d["image_id"]).zfill(12)}.jpg'),
                'question': d['question'],
                'label': d['answers'],
            })
    elif data_name == 'gqa':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, d['image_id'] + '.jpg'),
                'question': d['question'],
                'label': d['answers'],
            })
    elif data_name == 'aokvqa':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, f'COCO_val2014_{str(d["image_id"]).zfill(12)}.jpg'),
                'question': d['question'],
                'label': d['answers'],
            })
    elif data_name == 'vstar':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, d['image']),
                'question': d['text'],
                'label': d['label'],
                'short_question': d['text'].split('\n')[0],
            })
    elif data_name == 'docvqa':
        data_list = data_list['data']
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, d['image']),
                'question': d['question'],
                'label': d['answers'],
            })
    elif data_name == 'pope':
        for d in data_list:
            data.append({
                'image_path': os.path.join(image_path, d['image']),
                'question': d['text'],
                'label': d['label'],
            })
    
    return data