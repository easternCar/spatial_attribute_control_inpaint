# Spatial-aware Attribute Controllable GAN for Image Inpainting
A PyTorch reimplementation for the paper [SAC-GAN : Face Image Inpainting with Spatial-aware Attribute Controllable GAN]. (ACCV 2022)

## Abstract 
The objective of image inpainting is refilling the masked area with semantically appropriate pixels and producing visually realistic images as an output. After the introduction of generative adversarial networks (GAN), many inpainting approaches are showing promising development. Several attempts have been recently made to control reconstructed output with the desired attribute on face images using exemplar images and style vectors. Nevertheless, conventional style vector has the limitation that to project style attribute representation onto linear vector without preserving dimensional information. We introduce spatial-aware attribute controllable GAN (SAC-GAN) for face image inpainting, which is effective for reconstructing masked images with desired controllable facial attributes with advantage of utilizing style tensors as spatial forms. Various experiments to control over facial characteristics demonstrate the superiority of our method compared with previous image inpainting methods.

## Instruction
Will be updated soon........

Facial recognition model (https://drive.google.com/file/d/1HuDJDlQtpUzW62tj_wkaeAUPU96wC_D4/view?usp=sharing) which needs to be put in 'models' directory
This pre-trained model is trained on CASIA-WebFace and ArcFace using 128x128 image

## training

1. Prepare dataset

Put dataset images under 'archive' directory
---- 
We have small samples including smile/non-smile face datasets split into train and test dataset ()

2. Execute shell script file
----


## Inference


## Examples

(Table)

# Masks

We included one custom binary mask 'face_mask.png'.
As for random masks, we used mask generator fom DeepFill repository ()


## Acknowledgement
 + Most functions are brought from L2M-GAN(https://github.com/songquanpeng/L2M-GAN).
 + This work was supported by Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT)(No.2017-0-00897, Development of Object Detection and Recognition for Intelligent Vehicles) and (No.B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis)