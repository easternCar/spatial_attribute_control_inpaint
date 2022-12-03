# Spatial-aware Attribute Controllable GAN for Image Inpainting
A PyTorch reimplementation for the paper [SAC-GAN : Face Image Inpainting with Spatial-aware Attribute Controllable GAN]. (ACCV 2022)

## Abstract 
The objective of image inpainting is refilling the masked area with semantically appropriate pixels and producing visually realistic images as an output. After the introduction of generative adversarial networks (GAN), many inpainting approaches are showing promising development. Several attempts have been recently made to control reconstructed output with the desired attribute on face images using exemplar images and style vectors. Nevertheless, conventional style vector has the limitation that to project style attribute representation onto linear vector without preserving dimensional information. We introduce spatial-aware attribute controllable GAN (SAC-GAN) for face image inpainting, which is effective for reconstructing masked images with desired controllable facial attributes with advantage of utilizing style tensors as spatial forms. Various experiments to control over facial characteristics demonstrate the superiority of our method compared with previous image inpainting methods.


## Instruction

<p align="center"><img src="sample_imgs/example2.png" width="720"\></p>

Will be updated soon........

Facial recognition model (https://drive.google.com/file/d/1HuDJDlQtpUzW62tj_wkaeAUPU96wC_D4/view?usp=sharing) which needs to be put in 'models' directory
This pre-trained model is trained on CASIA-WebFace and ArcFace using 128x128 image

Download and put the checkpt file into 'models' directory.

## Example of dataset
1. We followed the datset settings from L2M-GAN(https://github.com/songquanpeng/L2M-GAN).
2. Prepare dataset should contains 'train' and 'test' folder
  - Class 1  
    - Train
        + image 1-1  
        + image 1-2
        ...    
        + image 1-n 
    - Test
        + image 1-1  
        + image 1-2
        ...    
        + image 1-n 
  - Class 2  
    - Train
        + image 2-1  
        + image 2-2
        ...    
        + image 2-n 
    - Test
        + image 2-1  
        + image 2-2
        ...    
        + image 2-n 

Example of dataset (small)

## training

Preparing....


## Inference

Preparing....

# Masks

We included one custom binary mask 'face_mask.png'.
As for random masks, we used mask generator fom DeepFill repository (https://github.com/zhaoyuzhi/deepfillv2)


## Acknowledgement
 + Most functions are brought from L2M-GAN(https://github.com/songquanpeng/L2M-GAN).
 + This work was supported by Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT)(No.2017-0-00897, Development of Object Detection and Recognition for Intelligent Vehicles) and (No.B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis)