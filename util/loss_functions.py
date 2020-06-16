import torch
import torch.nn as nn

def downsample(input, times):
    ds = nn.MaxPool2d((1, 1), (2, 2), (0, 0), ceil_mode=True)
    ds = ds.cuda()
    output = input.clone()

    for i in range(times):
        output = ds(output)

    return output

def gram_matrix(feature, s, GPUID):
    c,h,w = feature.size()
    x = feature.view(-1).cuda(GPUID)
    x = x[x.nonzero()]
    x = x.view(c, -1)
    x = torch.mm(x, x.t())
    return x

def whiten(cF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    # print(contentConv.size())
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)
    return whiten_cF

def cal_whiten_loss(target_img, res):
    whiten_loss = ((whiten(target_img.squeeze(0).view(3,-1).cpu().double()).float() - 
                    whiten(res.squeeze(0).view(3,-1).cpu().double()).float())**2).mean()
    return whiten_loss

def calc_mean_std(feat, GPUID, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_copy = feat.view(-1).cuda(GPUID)
    feat_copy = feat_copy[feat_copy.nonzero()]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def cal_cycle_loss(generated_features, generated_skips, img_mask, features, skips, vid_mask, GPUID, class_n, size, eps=1e-5):
    c1_loss = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    c2_loss = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    c = 0
    for k in range(class_n):
        if img_mask[k].max() == 1:
            c += 1
            gram_matrix_org = gram_matrix((features[0]+eps).squeeze(0)*vid_mask[k].cuda(GPUID), vid_mask[k].size(0), GPUID)
            gram_matrix_gen = gram_matrix((generated_features[0]+eps).squeeze(0)*img_mask[k].cuda(GPUID), img_mask[k].size(0), GPUID)
            c1_loss += ((gram_matrix_org - gram_matrix_gen)**2).mean().mean()

            img_mask_ds = downsample(img_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
            img_mask_ds = img_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
            vid_mask_ds = downsample(vid_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
            vid_mask_ds = vid_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
            gram_matrix_org = gram_matrix((features[1]+eps).squeeze(0)*vid_mask_ds, vid_mask_ds.size(0), GPUID)
            gram_matrix_gen = gram_matrix((generated_features[1]+eps).squeeze(0)*img_mask_ds, img_mask_ds.size(0), GPUID)
            c2_loss += ((gram_matrix_org - gram_matrix_gen)**2).mean().mean()
    c4_loss = ((features[2] - generated_features[2])**2).mean().mean()
    c1_skip_loss = ((skips[0][0].cuda(GPUID) - generated_skips[0][0])**2 + 
                    (skips[0][1].cuda(GPUID) - generated_skips[0][1])**2 + 
                    (skips[0][2].cuda(GPUID) - generated_skips[0][2])**2).mean().mean()
    c2_skip_loss = ((skips[1][0].cuda(GPUID) - generated_skips[1][0])**2 + 
                    (skips[1][1].cuda(GPUID) - generated_skips[1][1])**2 + 
                    (skips[1][2].cuda(GPUID) - generated_skips[1][2])**2).mean().mean()
    c3_skip_loss = ((skips[2][0].cuda(GPUID) - generated_skips[2][0])**2 + 
                    (skips[2][1].cuda(GPUID) - generated_skips[2][1])**2 + 
                    (skips[2][2].cuda(GPUID) - generated_skips[2][2])**2).mean().mean()

    c1_loss /= c
    c2_loss /= c

    return c1_loss, c2_loss, c4_loss, c1_skip_loss, c2_skip_loss, c3_skip_loss

def cal_flow_loss(org_frame_feature, gen_frame_feature, org_0, gen_0, img_mask, vid_mask, GPUID, class_n, seq_len, size):
    f1_loss = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    f2_loss = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    f1_long = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    f2_long = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    c = 0
    for k in range(class_n):
        if img_mask[k].max() == 1:
            c += 1
            for j in range(1,seq_len):

                g_flow = (gen_frame_feature[j][0]-gen_frame_feature[j-1][0]).squeeze(0).cuda(GPUID)
                o_flow = (org_frame_feature[j][0]-org_frame_feature[j-1][0]).squeeze(0).cuda(GPUID)
                g_flow_masked = (g_flow * img_mask[k].cuda(GPUID)).view(1, g_flow.size(0),-1)
                g_flow_mean = g_flow_masked.squeeze(0).mean(1)
                o_flow_masked = (o_flow * vid_mask[k].cuda(GPUID)).view(1, o_flow.size(0),-1)
                o_flow_mean = o_flow_masked.squeeze(0).mean(1)
                f1_loss += torch.abs(g_flow_mean - o_flow_mean).mean()

                img_mask_ds = downsample(img_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
                img_mask_ds = img_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
                vid_mask_ds = downsample(vid_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
                vid_mask_ds = vid_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
                g_flow = (gen_frame_feature[j][1]-gen_frame_feature[j-1][1]).squeeze(0).cuda(GPUID)
                o_flow = (org_frame_feature[j][1]-org_frame_feature[j-1][1]).squeeze(0).cuda(GPUID)
                g_flow_masked = (g_flow * img_mask_ds).view(g_flow.size(0),-1)
                g_flow_mean = g_flow_masked.squeeze(0).mean(1)
                o_flow_masked = (o_flow * vid_mask_ds).view(o_flow.size(0),-1)
                o_flow_mean = o_flow_masked.squeeze(0).mean(1)
                f2_loss += torch.abs(g_flow_mean - o_flow_mean).mean()

            g_flow = (gen_frame_feature[j][0]-gen_0[0]).squeeze(0).cuda(GPUID)
            o_flow = (org_frame_feature[j][0]-org_0[0]).squeeze(0).cuda(GPUID)
            g_flow_masked = (g_flow * img_mask[k].cuda(GPUID)).view(1, g_flow.size(0),-1)
            g_flow_mean = g_flow_masked.squeeze(0).mean(1)
            o_flow_masked = (o_flow * vid_mask[k].cuda(GPUID)).view(1, o_flow.size(0),-1)
            o_flow_mean = o_flow_masked.squeeze(0).mean(1)
            f1_long += torch.abs(g_flow_mean - o_flow_mean).mean()

            img_mask_ds = downsample(img_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
            img_mask_ds = img_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
            vid_mask_ds = downsample(vid_mask[k].view(1, size[0], size[1]), 1).cuda(GPUID)
            vid_mask_ds = vid_mask_ds[:, :int(size[0]/2), :int(size[1]/2)]
            g_flow = (gen_frame_feature[j][1]-gen_0[1]).squeeze(0).cuda(GPUID)
            o_flow = (org_frame_feature[j][1]-org_0[1]).squeeze(0).cuda(GPUID)
            g_flow_masked = (g_flow * img_mask_ds).view(g_flow.size(0),-1)
            g_flow_mean = g_flow_masked.squeeze(0).mean(1)
            o_flow_masked = (o_flow * vid_mask_ds).view(o_flow.size(0),-1)
            o_flow_mean = o_flow_masked.squeeze(0).mean(1)
            f2_long += torch.abs(g_flow_mean - o_flow_mean).mean()

    f1_loss /= c
    f2_loss /= c
    f1_long /= c
    f2_long /= c

    return f1_loss, f2_loss, f1_long, f2_long

def cal_style_loss(org_feat, gen_feat, vid_mask, img_mask, class_n, GPUID, size, eps=1e-5):
    loss = torch.tensor(0.0).type(torch.cuda.FloatTensor).cuda(GPUID)
    c = 0

    for i in range(4):
        for k in range(class_n):
            if img_mask[k].max() != 0:
                c += 1

                img_mask_ds = downsample(img_mask[k].view(1, size[0], size[1]), i).cuda(GPUID)
                img_mask_ds = img_mask_ds[:, :int(size[0]/(2**i)), :int(size[1]/(2**i))]
                vid_mask_ds = downsample(vid_mask[k].view(1, size[0], size[1]), i).cuda(GPUID)
                vid_mask_ds = vid_mask_ds[:, :int(size[0]/(2**i)), :int(size[1]/(2**i))]

                o_feat = (org_feat[i]+eps) * vid_mask_ds
                g_feat = (gen_feat[i]+eps) * img_mask_ds

                org_mean, org_std = calc_mean_std(o_feat, GPUID)
                gen_mean, gen_std = calc_mean_std(g_feat, GPUID)

                loss += ((org_mean.view(-1) - gen_mean.view(-1))**2).mean() + ((org_std.view(-1) - gen_std.view(-1))**2).mean()

    loss /= c
    return loss

def cal_perceptual_loss(org_feat, gen_feat, GPUID):
    perceptual_loss = torch.tensor([0.0]).type(torch.cuda.FloatTensor).cuda(GPUID)
    weight = 1
    for (o, g) in zip(org_feat, gen_feat):
        perceptual_loss += weight * (((torch.flip(o.cuda(GPUID), [3]).view(-1) 
            - g.cuda(GPUID).view(-1))**2).mean())
        weight += 1
    return perceptual_loss