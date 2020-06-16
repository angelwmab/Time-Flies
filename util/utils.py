import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from natsort import natsorted

def save_model(net, name):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
    torch.save(state_dict, name)

def make_video(target_img_pth, vid_dir, save_dir):
    gen_fileNames = os.listdir(save_dir)
    gen_fileNames = natsorted(gen_fileNames)
    org_fileNames = os.listdir(vid_dir)
    org_fileNames = natsorted(org_fileNames)

    target_img = cv2.imread(target_img_pth)
    gen_img = cv2.imread('%s%s'%(save_dir, gen_fileNames[0]))
    org_img = cv2.imread('%s%s'%(vid_dir, org_fileNames[0]))

    target_h, target_w, _ = target_img.shape
    gen_h, gen_w, _ = gen_img.shape
    org_h, org_w, _ = org_img.shape

    width = target_w + gen_w + org_w
    height = max(target_h, gen_h, org_h)
    result_image = np.zeros((height, width, 3), np.uint8)

    result_image[int((height-target_h)/2):int((height-target_h)/2)+target_h, :target_w] = target_img
    result_image[int((height-org_h)/2):int((height-org_h)/2)+org_h, target_w:target_w+org_w] = org_img
    result_image[int((height-gen_h)/2):int((height-gen_h)/2)+gen_h, target_w+org_w:] = gen_img

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter('%svideo.avi'%(save_dir), fourcc,25,(width,height))
    video.write(result_image)

    for org_f, gen_f in zip(org_fileNames, gen_fileNames):
        gen_img = cv2.imread('%s%s'%(save_dir, org_f))
        org_img = cv2.imread('%s%s'%(vid_dir, gen_f))
        result_image[int((height-target_h)/2):int((height-target_h)/2)+target_h, :target_w] = target_img
        result_image[int((height-org_h)/2):int((height-org_h)/2)+org_h, target_w:target_w+org_w] = org_img
        result_image[int((height-gen_h)/2):int((height-gen_h)/2)+gen_h, target_w+org_w:] = gen_img

        video.write(result_image)

    video.release()

def get_gaussian_filter():
    g_filter = np.array([[[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]],
                     [[2, 4 ,2],
                      [4, 8, 4],
                      [2, 4, 2]],
                     [[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]])
    g_filter = g_filter/np.sum(g_filter)
    g_filter_torch = torch.from_numpy(g_filter).unsqueeze(0)

    gaussian_filter = nn.Conv3d(1, 1, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
    gaussian_filter.weight.requires_grad = False
    gaussian_filter.weight.data = g_filter_torch.float().unsqueeze(0).expand(1, -1, -1, -1, -1)

    return gaussian_filter

def get_smooth_noise(h, w, length):
    gaussian_filter = get_gaussian_filter()
    smooth_noise = torch.randn((1, 1, length, h, w))/2 + 1
    smooth_noise = gaussian_filter(smooth_noise)
    smooth_noise = smooth_noise.squeeze(0).squeeze(0)

    return smooth_noise

def MaskHelper(seg,color):
    # green
    mask = torch.Tensor()
    if(color == 'background'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'building'):
        mask = torch.gt(seg[0], 0.5)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'ground'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 0.5))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'mountain'):
        mask = torch.gt(seg[0], 0.5)
        mask = torch.mul(mask,torch.gt(seg[1], 0.5))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'road'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 0.5))
    elif(color == 'sky'):
        mask = torch.gt(seg[0], 0.5)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 0.5))
    elif(color == 'tree'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 0.5))
        mask = torch.mul(mask,torch.gt(seg[2], 0.5))
    elif(color == 'water'):
        mask = torch.gt(seg[0], 0.5)
        mask = torch.mul(mask,torch.gt(seg[1], 0.5))
        mask = torch.mul(mask,torch.gt(seg[2], 0.5))
    
    else:
        print('MaskHelper(): color not recognized, color = ' + color)
    return mask.float()

def ExtractMask(Seg):
    # Given segmentation for content and style, we get a list of segmentation for each color
    color_codes = ['background', 'building', 'ground', 'mountain', 'road', 'sky', 'tree', 'water']
    masks = MaskHelper(Seg,color_codes[0]).unsqueeze(0).cuda()
    for color in color_codes[1:]:
        mask = MaskHelper(Seg,color).unsqueeze(0).cuda()
        masks = torch.cat((masks, mask), 0)
    return masks