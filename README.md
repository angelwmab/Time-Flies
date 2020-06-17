# Time Flies: Animating a Still Image with Time Lapse Video as Reference (CVPR 2020 accepted)
PyTorch implementaton of the following paper. In this paper, we propose a self-supervised end-to-end model to generate the timelapse
video from a single image and a reference video.  
<div align=><img height="200" src="https://github.com/angelwmab/Time-Flies/blob/master/figure/teaser.gif"/></div>

## Paper
[Time Flies: Animating a Still Image With Time-Lapse Video As Reference](http://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Time_Flies_Animating_a_Still_Image_With_Time-Lapse_Video_As_CVPR_2020_paper.pdf)  
Chia-Chi Cheng, Hung-Yu Chen, [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.  

Please cite our paper if you find it useful for your research.  
```
@InProceedings{Cheng_2020_CVPR,
author = {Cheng, Chia-Chi and Chen, Hung-Yu and Chiu, Wei-Chen},
title = {Time Flies: Animating a Still Image With Time-Lapse Video As Reference},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Example Results
![img](https://github.com/angelwmab/Time-Flies/blob/master/figure/qualitative.png)

## Installation
* This code was developed with Python 3.6.9 & Pytorch 1.0.0 & CUDA 10.1.
* Other requirements: cv2, numpy, natsort
* Clone this repo
```
git clone https://github.com/angelwmab/Time-Flies.git
cd Time-Flies-Animating-a-Still-Image-with-Time-Lapse-Video-as-Reference
```

## Testing
Download our pretrained models from [here](https://drive.google.com/open?id=1Jn_uE3U5aW8TAGcA_pEr79MaeDYiD5re) and put them under `models/`.  
Run the sample data provided in this repo:
```
python test.py
```
Run your own data:
```
python test.py --vid_dir YOUR_REF_VID_FRAME_PATH
               --seg_dir YOUR_SEGMENTATION_PATH
               --target_img_path YOUR_TARGET_IMG_PATH
```

## Training
Download the webcamclipart dataset [here](http://graphics.cs.cmu.edu/projects/webcamdataset/) and put them under `webcamclipart/`.  
Download the segmentation maps of each scene [here](https://drive.google.com/drive/folders/1_RGhDdLSpdrb_bk0x-EkXz9Jmhm3AQHY?usp=sharing) and put them under `segmentations/`.  
Then you can directly run the training code:
```
python train.py
```
If you want to train the model with your own dataset:
```
python train.py --vid_dir YOUR_REF_VID_DATASET
                --seg_dir YOUR_SEGMENTATION_DIR
```
