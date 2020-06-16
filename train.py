import os
import copy
import torch
import random

from model import Encoder, Decoder1, Decoder2

from util.utils import get_smooth_noise
from util.dataloader import folder_length_dict, get_seg_mask, get_ref_video_train
from util.loss_functions import cal_whiten_loss, cal_cycle_loss, cal_flow_loss, cal_style_loss, cal_perceptual_loss

VID_DIR = './webcamclipart/'
SEG_DIR = './segmentations/'
SAVE_DIR = './time_flies_models/'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_dir',type=str,help='directory of reference video frames',default=VID_DIR)
    parser.add_argument('--seg_dir',type=str,help='directory of segmentation maps',default=SEG_DIR)
    parser.add_argument('--save_dir',type=str,help='directory of models',default=SAVE_DIR)
    parser.add_argument('--gpu',type=int,default=[0,1,2],nargs=3)
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--class_n',type=int,help='segmentation classes',default=8)

    parser.add_argument('--dis_interval',type=int,default=10)
    parser.add_argument('--only_train_gen',type=int,default=2)
    parser.add_argument('--seq_len',type=int,default=3)
    parser.add_argument('--vid_len',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--dis_lr',type=float,default=2e-5)

    parser.add_argument('--alpha',type=float,default=1)
    parser.add_argument('--perceptual_weight',type=float,default=0.1)
    parser.add_argument('--adv_weight',type=float,default=1)
    parser.add_argument('--cycle_weight',type=float,default=0.05)
    parser.add_argument('--content_weight',type=float,default=0.05)
    parser.add_argument('--style_weight',type=float,default=10)
    parser.add_argument('--flow_weight',type=float,default=100)

    args = parser.parse_args()
    return args

args = get_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(args.gpu[0])

# models
encoder = Encoder(args.seq_len).cuda(args.gpu[0])
decoder1 = Decoder1(args.gpu[1]).cuda(args.gpu[1])
decoder2 = Decoder2(args.gpu[2]).cuda(args.gpu[2])
real_smooth_discriminator = Realness_Smoothness_discriminator(args.seq_len).cuda(args.gpu[0])
print('-----finish building models-----')

# optimizers
encoder_optimizer = torch.optim.Adam(encoder.vid_first.parameters(), lr=args.lr)
fine_tune_params = list(encoder.vid_enc_1.parameters()) + list(encoder.vid_enc_2.parameters())
vgg_optimizer = torch.optim.SGD(fine_tune_params, lr=0.0001, momentum=0.9, weight_decay=1e-4)
decoder_params = list(decoder1.parameters()) + list(decoder2.parameters())
decoder_optimizer = torch.optim.Adam(decoder_params, lr=args.lr)
real_smooth_discriminator_optimizer = torch.optim.Adam(real_smooth_discriminator.parameters(), lr=args.dis_lr, betas=(0.5, 0.99))
print('-----finish setting optimizers-----')

# get the length of each training video
inside_folders, folder_length = folder_length_dict(args.vid_dir)

data_count = 0
dis_interval = args.dis_interval
alpha = args.alpha
for epoch in range(args.epochs):
    for f in inside_folders:
        ref_vid = get_ref_video_train(args.vid_dir, f, folder_length[f], args.vid_len, args.size)
        target_img = ref_vid.mean(0).unsqueeze(0)
        target_img = torch.flip(target_img, [3])
        vid_mask = get_seg_mask('%s%s.png'%(args.seg_dir, f), args.size)
        img_mask = get_seg_mask('%s%s.png'%(args.seg_dir, f), args.size, flip=True)

        ref_vid.cuda(args.gpu[0])
        target_img.cuda(args.gpu[0])

        # Compute the first 3 frames (depends on seq_len)
        for frame in range(seq_len):
            # make vid_seq
            ref_vid_seq = ref_vid[0][:][:][:]
            for _ in range(seq_len-frame-1):
                ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[0][:][:][:]), 0)
            for i in range(1,frame+1):
                ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[i][:][:][:]), 0)
            ref_vid_seq = ref_vid_seq.unsqueeze(0).cuda(args.gpu[0])

            # encode
            features, skips = encoder(target_img, ref_vid_seq)

            # clone features to another gpu
            features_clone = list()
            for feat in features[:3]:
                features_clone.append(feat.cuda(args.gpu[1]))
            skips_clone = list()
            for skip in skips:
                skips_clone.append([skip[0].cuda(args.gpu[1]), skip[1].cuda(args.gpu[1]), skip[2].cuda(args.gpu[1])])

            # decode
            if frame == 0:
                h, w = features[0].size()[2:]
                smooth_noise = get_smooth_noise(h, w, args.vid_len)
                output = decoder1(features_clone, skips_clone, img_mask.cuda(args.gpu[1]), vid_mask.cuda(args.gpu[1]))
                result = [decoder2(output.cuda(args.gpu[2]), features_clone[0].cuda(args.gpu[2]), img_mask.cuda(args.gpu[2]), vid_mask.cuda(args.gpu[2]), noise=smooth_noise[frame].cuda(args.gpu[2]))]
            else:
                output = decoder1(features_clone, skips_clone, img_mask.cuda(args.gpu[1]), vid_mask.cuda(args.gpu[1]))
                result.append(decoder2(output.cuda(args.gpu[2]), features_clone[0].cuda(args.gpu[2]), img_mask.cuda(args.gpu[2]), vid_mask.cuda(args.gpu[2]), noise=smooth_noise[frame].cuda(args.gpu[2])))

        # record the first frame of gen. seq. & org. seq. (for long flow loss)
        org_0 = encoder(ref_vid[0][:][:][:].unsqueeze(0).cuda(args.gpu[0]))
        gen_0 = encoder(result[0].cuda(args.gpu[0]))

        # cycle loss & content loss
        generated_vid_seq = result[0]
        for i in range(1, seq_len):
            generated_vid_seq = torch.cat((generated_vid_seq, result[i]), 1)
        generated_features, generated_skips = encoder(result[-1].cuda(args.gpu[0]), generated_vid_seq.cuda(args.gpu[0]))

        c1_loss, c2_loss, c4_loss, c1_skip_loss, c2_skip_loss, c3_skip_loss = cal_cycle_loss(generated_features, generated_skips, img_mask, features, skips, vid_mask, args.gpu[0], args.class_n, args.size)
        whiten_loss = cal_whiten_loss(target_img, result[-1])

        cycle_loss = (c1_loss + c2_loss) * args.cycle_weight
        content_loss = (args.alpha * (c4_loss + c1_skip_loss + c2_skip_loss + c3_skip_loss) + \
                        (1 - args.alpha) * whiten_loss.cuda(args.gpu[0])) * args.content_weight

        # flow loss
        org_frame_feature = list()
        gen_frame_feature = list()
        for i in range(args.seq_len):
            org_frame_feature.append(encoder((video[i][:][:][:]).unsqueeze(0).cuda(args.gpu[0])))
            gen_frame_feature.append(encoder(result[i].cuda(args.gpu[0])))

        f1_loss, f2_loss, f1_long, f2_long = cal_flow_loss(org_frame_feature, gen_frame_feature, org_0, gen_0, img_mask, vid_mask, args.gpu[0], args.class_n, args.seq_len, args.size)
        flow_loss = (f1_loss + f2_loss + f1_long + f2_long) * args.flow_weight

        # adversarial loss
        output = real_smooth_discriminator(ref_vid_seq.cuda(args.gpu[0]), generated_vid_seq.cuda(args.gpu[0]))
        adv_loss = -args.adv_weight * torch.log(output)

        # perceptual loss
        perceptual_loss = args.perceptual_weight * cal_perceptual_loss(org_frame_feature[0], gen_frame_feature[0], args.gpu[0])

        # style loss (from AdaIN)
        style_loss = args.style_weight * cal_style_loss(org_frame_feature[0], gen_frame_feature[0], vid_mask, img_mask, args.class_n, args.gpu[0], args.size)

        # total loss
        total_loss = cycle_loss + content_loss + adv_loss + flow_loss + perceptual_loss + style_loss
        vid_loss = cycle_loss + adv_loss + flow_loss + perceptual_loss + style_loss

        # back-prop
        decoder_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        decoder_optimizer.step()

        encoder_optimizer.zero_grad()
        vgg_optimizer.zero_grad()
        vid_loss.backward(retain_graph=True)
        encoder_optimizer.step()
        vgg_optimizer.step()

        data_count += 1

        # Compute other frames
        for frame in range(args.seq_len, args.vid_len):
            for i in range(seq_len - 1):
                result[i] = result[i + 1]
            ref_vid_seq = ref_vid[frame-seq_len+1][:][:][:]
            for i in range(frame-seq_len+2, frame+1):
                ref_vid_seq = torch.cat((ref_vid_seq, ref_vid[i][:][:][:]), 0)
            ref_vid_seq = ref_vid_seq.unsqueeze(0).cuda(args.gpu[0])

            # encode
            features, skips = encoder(target_img, ref_vid_seq)

            # clone features to another gpu
            features_clone = list()
            for feat in features[:3]:
                features_clone.append(feat.cuda(args.gpu[1]))
            skips_clone = list()
            for skip in skips:
                skips_clone.append([skip[0].cuda(args.gpu[1]), skip[1].cuda(args.gpu[1]), skip[2].cuda(args.gpu[1])])

            # decode
            output = decoder1(features_clone, skips_clone, img_mask.cuda(args.gpu[1]), vid_mask.cuda(args.gpu[1]))
            result[args.seq_len-1] = decoder2(output.cuda(args.gpu[2]), features_clone[0].cuda(args.gpu[2]), img_mask.cuda(args.gpu[2]), vid_mask.cuda(args.gpu[2]), noise=smooth_noise[frame].cuda(args.gpu[2]))
            
            # cycle loss & content loss
            generated_vid_seq = result[0]
            for i in range(1, seq_len):
                generated_vid_seq = torch.cat((generated_vid_seq, result[i]), 1)
            generated_features, generated_skips = encoder(result[-1].cuda(args.gpu[0]), generated_vid_seq.cuda(args.gpu[0]))

            c1_loss, c2_loss, c4_loss, c1_skip_loss, c2_skip_loss, c3_skip_loss = cal_cycle_loss(generated_features, generated_skips, img_mask, features, skips, vid_mask, args.gpu[0], args.class_n, args.size)
            whiten_loss = cal_whiten_loss(target_img, result[-1])

            cycle_loss = (c1_loss + c2_loss) * args.cycle_weight
            content_loss = (args.alpha * (c4_loss + c1_skip_loss + c2_skip_loss + c3_skip_loss) + \
                            (1 - args.alpha) * whiten_loss.cuda(args.gpu[0])) * args.content_weight

            # flow loss
            org_frame_feature = list()
            gen_frame_feature = list()
            for i in range(args.seq_len):
                org_frame_feature.append(encoder((video[frame-args.seq_len+i][:][:][:]).unsqueeze(0).cuda(args.gpu[0])))
                gen_frame_feature.append(encoder(result[i].cuda(args.gpu[0])))

            f1_loss, f2_loss, f1_long, f2_long = cal_flow_loss(org_frame_feature, gen_frame_feature, org_0, gen_0, img_mask, vid_mask, args.gpu[0], args.class_n, args.seq_len, args.size)
            flow_loss = (f1_loss + f2_loss + f1_long + f2_long) * args.flow_weight

            # adversarial loss
            output = real_smooth_discriminator(ref_vid_seq.cuda(args.gpu[0]), generated_vid_seq.cuda(args.gpu[0]))
            adv_loss = -args.adv_weight * torch.log(output)

            # perceptual loss
            perceptual_loss = args.perceptual_weight * cal_perceptual_loss(org_frame_feature[0], gen_frame_feature[0], args.gpu[0])

            # style loss (from AdaIN)
            style_loss = args.style_weight * cal_style_loss(org_frame_feature[0], gen_frame_feature[0], vid_mask, img_mask, args.class_n, args.gpu[0], args.size)

            # total loss
            total_loss = cycle_loss + content_loss + adv_loss + flow_loss + perceptual_loss + style_loss
            vid_loss = cycle_loss + adv_loss + flow_loss + perceptual_loss + style_loss

            # back-prop
            decoder_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            vgg_optimizer.zero_grad()
            vid_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            vgg_optimizer.step()

            data_count += 1

            # train discriminator
            if epoch+1 >= args.only_train_gen and data_count % dis_interval == 0:
                real_output = real_smooth_discriminator(ref_vid_seq.detach().cuda(args.gpu[0]), torch.flip(ref_vid_seq.detach(), [3]).cuda(args.gpu[0]))
                fake_output = real_smooth_discriminator(ref_vid_seq.detach().cuda(args.gpu[0]), generated_vid_seq.detach().cuda(args.gpu[0]))
                d_loss = -(torch.log(real_output) + torch.log(1 - fake_output))
                real_smooth_discriminator_optimizer.zero_grad()
                d_loss.backward()
                real_smooth_discriminator_optimizer.step()

    if dis_interval < vid_length:
        dis_interval += 1

    if alpha > 0.8:
        alpha *= 0.975

    print('------ epoch %d ------'%(epoch))
    print('----- save model -----')
    save_model(encoder, '%sencoder_%d.pth'%(args.save_dir, epoch))
    save_model(decoder1, '%sdecoder1_%d.pth'%(args.save_dir, epoch))
    save_model(decoder2, '%sdecoder2_%d.pth'%(args.save_dir, epoch))

print('-----finish training-----')