import torch
import os
import random
from natsort import natsorted
from torchvision import transforms
from util.utils import ExtractMask

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def folder_length_dict(vid_dir):
    inside_folders = os.listdir(vid_dir)

    folder_length = dict()
    for f in inside_folders:
        path = vid_dir + f + '/'
        folder_length[f] = len(os.listdir(path))

    return inside_folders, folder_length

def get_seg_mask(seg_path, size, flip=False):
    if flip:
        transform = transforms.Compose(
                [transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor()])
    else:
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

def get_ref_video_train(vid_dir, folder, folder_length, vid_length, size):
    vid_frame_paths = os.listdir(vid_dir + folder + '/')
    vid_frame_paths = natsorted(vid_frame_paths)

    # randomly choose a video clip
    start_frame = random.randint(0, folder_length - vid_length)

    img_transform = transforms.Compose(
                [transforms.Resize(size),
                transforms.ToTensor()])

    path = vid_dir + folder + '/' + vid_frame_paths[start_frame]
    video = Image.open(path).convert('RGB')
    video = img_transform(video)
    video = video.unsqueeze(0)

    for i in range(start_frame + 1, start_frame + vid_length):
        path = vid_dir + folder + '/' + vid_frame_paths[i]
        img = Image.open(path).convert('RGB')
        img = img_transform(img)
        img = img.unsqueeze(0)

        video = torch.cat((video, img), 0)

    return video

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