import torch
import random
import os
import cv2
import numpy as np


def load_predefined_mask_noshuffle(args, batch_size, chose_idx):
    if args.mask_type != 'RAND':
        return

    img_height = img_width = args.img_size
    masks = []

    mask_files = os.listdir(args.rand_mask_path)
    mask_files = sorted(mask_files)
    MASK_NUM = len(mask_files)
    
    chosen = []
    for j in range(batch_size):
        chosen.append(chose_idx)
        chose_idx = chose_idx + 1

        if chose_idx >= MASK_NUM:
            chose_idx = 0
        
    for i in range(batch_size):
        BINARY_MASK = cv2.imread(args.rand_mask_path + '/' + mask_files[chosen[i]], cv2.IMREAD_GRAYSCALE)
        w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]
        
        if w <img_width or w > img_width or h > img_height or h < img_height:
            BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        # get mask
        thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
        masks.append(BINARY_MASK)
 
    return masks, chose_idx


    

def load_predefined_mask(args, batch_size):
    img_height = img_width = args.img_size
    
    masks = []

    # if FIX
    if args.mask_type == 'FIX':
        mask_file = args.fix_mask_path
        
        BINARY_MASK = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]

        if w <img_width or w > img_width or h > img_height or h < img_height:
            BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        # get mask
        thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
        for i in range(batch_size):
            masks.append(BINARY_MASK)

        

    # if rand
    elif args.mask_type == 'RAND':
        mask_files = os.listdir(args.rand_mask_path)
        mask_files = sorted(mask_files)
        MASK_NUM = len(mask_files)

        chosen = random.sample(range(0, MASK_NUM), batch_size)

        for i in range(batch_size):
            BINARY_MASK = cv2.imread(args.rand_mask_path + '/' + mask_files[chosen[i]], cv2.IMREAD_GRAYSCALE)
            w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]
            
            if w <img_width or w > img_width or h > img_height or h < img_height:
                BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

            # get mask
            thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
            masks.append(BINARY_MASK)
 
    return masks



# call when using predefined mask
def mask_image_predefined(args, x, masks):

    img_height = img_width = args.img_size

    # ------- masks : list of cv2 ndarray
    batch_size = len(masks)

    mask = torch.zeros((batch_size, 1, img_height, img_width), dtype=torch.float32)

    for i in range(batch_size):
        mask[i, :, :, :] = torch.Tensor(masks[i]) / 255.0

    if x.is_cuda:
        mask = mask.cuda()
        
    result = x * (1. - mask) #first, occlude with big mask

    return result, mask