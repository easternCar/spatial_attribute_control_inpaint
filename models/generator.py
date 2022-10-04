import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ResBlk, AdainResBlk


class Inpaint_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size, max_conv_dim, style_dim, w_hpf = args.img_size, args.g_max_conv_dim, args.style_dim, args.w_hpf
        self.first_dim_in = 2 ** 14 // img_size
        
        #dim_in = 2 ** 14 // img_size
        dim_in =self.first_dim_in
    
        # ------- my init
        self.from_rgb = nn.Conv2d(3+2, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.bottleneck = nn.ModuleList()

        # --------- my encoder
        dim_in =self.first_dim_in
        #if w_hpf > 0:
        #    repeat_num += 1
        for _ in range(3):
            dim_out = min(dim_in * 2, max_conv_dim)
            #print("EN " + str(_) + "========<" + str(dim_out) + "> <" + str(dim_in) + ">")
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, down_sample=True))
            dim_in = dim_out


        # ---------- my bottleneck in this, use pure style [0] (8x8)
        for _ in range(2):
            #self.encode.append(
            self.bottleneck.append(
                ResBlk(dim_out, dim_out, normalize=True))



    # x : image
    # s : style maps
    # occ_mask : binary mask
    def forward(self, x, occ_mask, masks=None):
        
        # first
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3)).cuda()
        x = torch.cat([x, ones, occ_mask], dim=1)
        x = self.from_rgb(x)

        # encode
        cache = {}
        skip_connect = []       # for skip_connection
        for bidx, block in enumerate(self.encode):
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
        
            x = block(x)    # <conv>
            #print(str(bidx)  + " [ENC] --x : " + str(x.size()))

            skip_connect += [x]     # in 0504, we save all feature


        # ---- bottleneeck
        for bidx, block in enumerate(self.bottleneck):
            x = block(x)


        # return feature x (Bx512x16x16), skip ([256x64x64, 512x32x32, 512x16x16])
        return x, skip_connect




class Inpaint_Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        img_size, max_conv_dim, style_dim, w_hpf = args.img_size, args.g_max_conv_dim, args.style_dim, args.w_hpf
        self.first_dim_in = 2 ** 14 // img_size
    
        if self.args.model_type == 'COMOD':
            ADAIN_TYPE = '1D'
        elif self.args.model_type == 'SAC':
            ADAIN_TYPE = '2D'

        #dim_in = 2 ** 14 // img_size
        dim_in =self.first_dim_in
        # ------- my init
        self.decode = nn.ModuleList()
        self.attmaps = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # 06/03
        self.last_layer = AdainResBlk(dim_in, dim_in, style_dim,
                               w_hpf=w_hpf, up_sample=False, use_type=ADAIN_TYPE)

        # ---------- my decoder
        # because of style and ADAIN, watch for size!
        # style_maps : [8] --> [0,1] 8x8, [2,3] 16x16, [4,5] 32x32, [6,7] 64x64,
        dim_out =self.first_dim_in
        for i in range(3):
            dim_in = min(dim_out * 2, max_conv_dim)
            #print("DE " + str(i) + "========<" + str(dim_out) + "> <" + str(dim_in) + ">")
            
            self.attmaps.insert(0, nn.Conv2d(dim_in, 1, 3, 1, 1))
            self.decode.insert(0, 
                AdainResBlk(dim_in*2, dim_out, style_dim,           # orig : dim_in (no skip)
                               w_hpf=w_hpf, up_sample=True, use_type=ADAIN_TYPE))  # stack-like
            #self.attmaps.insert(0, gen_conv(dim_in, 1, 3, 1, 1))

            dim_out = dim_in

        for _ in range(2):      # bottleneck part
            self.attmaps.insert(0, nn.Conv2d(dim_out, 1, 3, 1, 1))
            self.decode.insert(0,
                AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, use_type=ADAIN_TYPE))
            

        self.leakyRelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()


    # x : feature map from encoder
    # s : style maps
    # skip_connect : skip connection features from encoder
    def forward(self, x, s, skip_connects, masks=None):
        #s = s[0]

        #for sidx, stmap in enumerate(s):
        #    print("STYLE " + str(sidx) + " -- : " + str(stmap.size()))

        

        # x : [B, 512, 8, 8]
        #for block in self.decode:
        for bidx, block in enumerate(self.decode):
            
            
            att = self.sigmoid(self.attmaps[bidx](x)) # get attmap

            #s_map = s[0] if bidx < 2 else s[bidx-2]
            #print(str(bidx)  + " [DEC] --x : " + str(x.size()) + ",, s : " + str(s[sidx].size())+ ",, att : " + str(att.size()))
            
            # s_map = att * s_map   # ---we noted as 'noatt'
            
            if bidx < 2:
                sidx = 0
            else:
                sidx = bidx - 2
                skip_idx = 2 - (bidx - 2)
                x = torch.cat([x, skip_connects[skip_idx]], dim=1)

            s_i = s[sidx] # * att 
            x = block(x, s_i)  # <conv>

        x = self.last_layer(x, s[len(s)-1])
        return self.to_rgb(x)





