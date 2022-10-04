import copy
from munch import Munch
from models.generator import Inpaint_Encoder, Inpaint_Decoder       # GEN
from models.discriminator import Discriminator                      # DIS
from models.resnet18 import ResNet18                                # Perceptual
from models.modulation_mapper import ModMapper                      # Mapper
import torch.nn as nn
from models.resnet50_fr import CBAMResNet as CBAMResNet128_50       # Identity

def build_model(args):

    discriminator = nn.DataParallel(Discriminator(args), device_ids=args.visible_gpus)

    inpaint_encoder = nn.DataParallel(Inpaint_Encoder(args), device_ids=args.visible_gpus)
    inpaint_decoder = nn.DataParallel(Inpaint_Decoder(args), device_ids=args.visible_gpus)
    face_recognizer = None

    inpaint_encoder_ema = copy.deepcopy(inpaint_encoder)
    inpaint_decoder_ema = copy.deepcopy(inpaint_decoder)

    spatial_mapper = nn.DataParallel(ModMapper(args), device_ids=args.visible_gpus)
    spatial_mapper_ema = copy.deepcopy(spatial_mapper)
        

    nets = Munch(inpaint_encoder=inpaint_encoder, inpaint_decoder=inpaint_decoder,
                discriminator=discriminator,
                spatial_mapper=spatial_mapper)
    nets_ema = Munch(inpaint_encoder=inpaint_encoder_ema, inpaint_decoder=inpaint_decoder_ema,
                    spatial_mapper=spatial_mapper_ema)
                     

    # ------- perceptual
    resnet18 = ResNet18(args)
    nets.resnet18 = resnet18
    nets_ema.resnet18 = resnet18

    # ------- identity
    recognizer = CBAMResNet128_50(50, feature_dim=256, mode='ir')
    recognizer.load_pretrained()
    nets.recognizer = recognizer
    nets_ema.recognizer = recognizer

    return nets, nets_ema
