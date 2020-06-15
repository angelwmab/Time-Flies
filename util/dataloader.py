import torch
import os
import random
from util.utils import *
from natsort import natsorted
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_seg_mask(seg_path, size):
    transform = transforms.Compose(
                [transforms.Resize(size),
                transforms.ToTensor()])

    seg_map = Image.open(seg_path).convert('RGB')
    seg_map = transform(seg_map)
    seg_mask = ExtractMask(seg_map)

    return seg_mask

def get_target_img(target_img_path, size):
    transform = transforms.Compose(
                [transforms.Resize(size),
                transforms.ToTensor()])

    target_img = Image.open(target_img_path).convert('RGB')
    target_img = transform(target_img)
    target_img = target_img.unsqueeze(0)

    return target_img

def get_ref_video_test(vid_dir, size):
    vid_frame_paths = os.listdir(vid_dir)
    vid_frame_paths = natsorted(vid_frame_paths)

    vid_length = len(vid_frame_paths)
    start_frame = 0

    img_transform = transforms.Compose(
                [transforms.Resize(size),
                transforms.ToTensor()])

    path = vid_dir + vid_frame_paths[start_frame]
    video = Image.open(path).convert('RGB')
    video = img_transform(video)
    video = video.unsqueeze(0)

    for i in range(start_frame + 1, start_frame + vid_length):
        path = vid_dir + vid_frame_paths[i]
        img = Image.open(path).convert('RGB')
        img = img_transform(img)
        img = img.unsqueeze(0)

        video = torch.cat((video, img), 0)

    return video, vid_length