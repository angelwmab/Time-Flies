import os
import torch
import argparse
from PIL import Image
from torchvision.utils import save_image

from model import Encoder, Decoder1, Decoder2
from util.utils import make_video, get_smooth_noise
from util.dataloader import get_seg_mask, get_target_img, get_ref_video_test

VID_DIR = './samples/reference_video_frames/'
SEG_DIR = './samples/segmentations/'
MODEL_DIR = './models/'
SAVE_DIR = './samples/results/'
TARGET_IMG_PATH = './samples/target.jpg'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_dir',type=str,help='directory of reference video frames',default=VID_DIR)
    parser.add_argument('--seg_dir',type=str,help='directory of segmentation maps',default=SEG_DIR)
    parser.add_argument('--model_dir',type=str,help='directory of models',default=MODEL_DIR)
    parser.add_argument('--save_dir',type=str,help='directory of results',default=SAVE_DIR)
    parser.add_argument('--target_img_path',type=str,help='path to input target image',default=TARGET_IMG_PATH)
    parser.add_argument('--gpu',type=int, default=0)
    parser.add_argument('--size',type=int, default=512)

    args = parser.parse_args()
    return args

args = get_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(args.gpu)

seq_len = 3

encoder = Encoder(seq_len).cuda()
decoder1 = Decoder1(args.gpu).cuda()
decoder2 = Decoder2(args.gpu).cuda()
encoder.load_state_dict(torch.load('%sencoder.pth'%(args.model_dir)))
decoder1.load_state_dict(torch.load('%sdecoder1.pth'%(args.model_dir)))
decoder2.load_state_dict(torch.load('%sdecoder2.pth'%(args.model_dir)))
encoder.eval()
decoder1.eval()
decoder2.eval()
print('-----finish building models-----')

with torch.no_grad():
    ref_vid, vid_len = get_ref_video_test(args.vid_dir, args.size)
    ref_vid = ref_vid.cuda()
    target_img = get_target_img(args.target_img_path, args.size)
    target_img = target_img.cuda()
    vid_mask = get_seg_mask('%svid_seg.png'%(args.seg_dir), args.size)
    vid_mask = vid_mask.cuda()
    img_mask = get_seg_mask('%simg_seg.png'%(args.seg_dir), args.size)
    img_mask = img_mask.cuda()
    print('-----finish loading data-----')

    # Compute the first 3 frames (depends on seq_len)
    for frame in range(seq_len):
        ref_vid_seq = ref_vid[0][:][:][:]
        for _ in range(seq_len-frame-1):
            ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[0][:][:][:]), 0)
        for i in range(1,frame+1):
            ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[i][:][:][:]), 0)
        ref_vid_seq = ref_vid_seq.unsqueeze(0).cuda()

        features, skips = encoder(target_img, ref_vid_seq)
        if frame == 0:
            h, w = features[0].size()[2:]
            smooth_noise = get_smooth_noise(h, w, vid_len)

        res = decoder1(features, skips, img_mask, vid_mask)
        res = decoder2(res, features[0], img_mask, vid_mask, noise=smooth_noise[frame])
        save_image(res, '%s%d.jpg'%(args.save_dir, frame))

    # Compute other frames
    for frame in range(seq_len, vid_len):
        ref_vid_seq = ref_vid[frame-seq_len+1][:][:][:]
        for i in range(frame-seq_len+2, frame+1):
            ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[i][:][:][:]), 0)
        ref_vid_seq = ref_vid_seq.unsqueeze(0).cuda()

        features, skips = encoder(target_img, ref_vid_seq)
        res = decoder1(features, skips, img_mask, vid_mask)
        res = decoder2(res, features[0], img_mask, vid_mask, noise=smooth_noise[frame])
        save_image(res, '%s%d.jpg'%(args.save_dir, frame))

    print('-----finish frames generation-----')

    make_video(args.target_img_path ,args.vid_dir, args.save_dir)
    print('-----finish video generation-----')