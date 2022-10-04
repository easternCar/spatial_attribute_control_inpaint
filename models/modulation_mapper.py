
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------ my custom
class ModMapper(nn.Module):
    def __init__(self, args, w_size=[128, 16, 16], f_size=[512, 16, 16]):
        super().__init__()

        c_dim = args.num_domains
        self.num_domains = args.num_domains
        self.w_size = w_size
        self.args = args

        # layer (FC : c->W)
        self.fc1 = nn.Linear(c_dim, w_size[0])
        self.fc2 = nn.Linear(w_size[0], w_size[0] * w_size[1] * w_size[2])

        # mapp
        layers = []
        #layers += [nn.Linear(style_dim, style_dim)]
        layers += [nn.Conv2d(w_size[0] + f_size[0], w_size[0], 3, 1, 1)]        # first
        layers += [nn.ReLU()]

        for _ in range(3):
            #layers += [nn.Linear(style_dim, style_dim)]
            layers += [nn.Conv2d(w_size[0], w_size[0], 3, 1, 1)]
            layers += [nn.ReLU()]
        self.spatial_mapper = nn.Sequential(*layers)

        # resize convs
        self.resize_convs = nn.ModuleList()
        for _ in range(4):      # before 06/03, (3) 16->32->64,
            self.resize_convs += [nn.Conv2d(w_size[0], w_size[0], 3, 1, 1)]

        # -------- CROSS ATT
        if self.args.cross_attention == True:
            self.CrossAtt = cross_attention(args)

    # x : [B, 512, 16, 16] from decoder
    def forward(self, x, c):

        c = F.one_hot(c, num_classes=self.num_domains).float()

        h = self.fc1(c)
        h = self.fc2(h)

        batch_size = c.size()[0]

        # reshape
        h = h.reshape(batch_size, self.w_size[0], self.w_size[1], self.w_size[2])
        # normalize
        h = normalize_2nd_moment(h)     # h : B 512 16 16
        h = torch.cat([h, x], dim=1)    # h = [B 1024 16 16]
        
        s = self.spatial_mapper(h)      # h : B 512 16 16

        # ------------ CROSS ATT
        if self.args.cross_attention == True:
            s = self.CrossAtt(s, x)


        # array of style maps
        s_maps = []
        s_maps += [s]

        for resize_layer in self.resize_convs:
            s = resize_layer(s) 
            s_maps += [s]
            s = F.interpolate(s, scale_factor=2, mode='nearest')    #s_tilde

        return s_maps
        


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    #return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

    return x * (x.pow(2).mean(dim=dim, keepdim=True) + eps).rsqrt()


class cross_attention(nn.Module):
    def __init__(self, args, w_size=[128, 16, 16], f_size=[512, 16, 16]):
        super().__init__()

        self.s_size = w_size
        self.f_size = f_size

        self.f_conv_1 = nn.Conv2d(f_size[0], f_size[0] // 2, 3, 1, 1)
        self.f_conv_2 = nn.Conv2d(f_size[0] // 2, w_size[0], 3, 1, 1)
        self.s_conv_att = nn.Conv2d(w_size[0], w_size[0] // 8, 1, 1, 0)
        self.f_conv_att = nn.Conv2d(w_size[0], w_size[0] // 8, 1, 1, 0)

        self.s_conv_val = nn.Conv2d(w_size[0], w_size[0], 1, 1, 0)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    # s : [B, 128, 16, 16] from style
    # x : [B, 512, 16, 16] from decoder
    def forward(self, s, x):

        batch_size = s.size()[0]

        """
            inputs :
                s : [B, 128, 16, 16] from style
                x : [B, 512, 16, 16] -> [B, 128, 16, 16] from decoder
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """

        h = self.f_conv_1(x)
        h = self.f_conv_2(h)

        # proj_x -> [B, 16x16, 16], # proj_x -> [B, 16, 16x16]
        proj_x = self.f_conv_att(h).view(batch_size, -1, self.s_size[1] * self.s_size[2]).permute(0,2,1) # B X CX(N)
        proj_s = self.s_conv_att(s).view(batch_size, -1, self.s_size[1] * self.s_size[2])

        # attmap -> [B, 16x16, 16x16], proj_value -> [B, 16, 16x16]
        attmap = torch.bmm(proj_x, proj_s)
        attmap = self.softmax(attmap) # B X (N) X (N) 
        proj_value = self.s_conv_val(s).view(batch_size, -1, self.s_size[1] * self.s_size[2]) # B X C X N

        out = torch.bmm(proj_value, attmap.permute(0,2,1))      # [B, 16x16, 16]
        out = out.view(batch_size, self.s_size[0], self.s_size[1], self.s_size[2])

        out = self.gamma*out + s

        return out


