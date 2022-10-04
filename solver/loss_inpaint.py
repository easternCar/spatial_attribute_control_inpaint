import torch
import torch.nn as nn
from munch import Munch
import torch.nn.functional as F


# D)
# 1) put masked_x into style_encoder and get S(c0)
# 2) change S(c0) to S(c1)
# 3) get reg1, adv (real) 

# 4) generate fake image with S(c0)
# 5) get adv (fake)

# ------------
# 0~domains [0, 1, 2]...
# [0] : just 'inpaint'
# [1] : [0] -> [1] translation
# [2] : [0] -> [2] translation

def compute_d_loss(nets, args, masked_x, sample_org, sample_ref, masks=None, occ_mask=None):
    # Real images
    sample_org.x.requires_grad_()
    out = nets.discriminator(sample_org.x, sample_org.y)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, sample_org.x)


    # pick common feature for 0, 1, 2, ...
    with torch.no_grad():
        x_f, x_f_skips = nets.inpaint_encoder(x=masked_x, occ_mask=occ_mask)       # encoder (and output -> style encoder, decoder)
        target_s_map = nets.spatial_mapper(x_f, sample_ref.y)       # SAC : [s[0], s[1], ...],  COMOD : [s]
        x_fake = nets.inpaint_decoder(x=x_f, s=target_s_map, skip_connects=x_f_skips)    # G(x, target)

    # adv loss (fake)
    out = nets.discriminator(x_fake, sample_ref.y)
    loss_fake = adv_loss(out, 0)

    # image to image
    x_fake = x_fake * occ_mask + sample_org.x * (1. - occ_mask) # improve
    global_fake_out = nets.discriminator(x_fake, sample_ref.y)  # improve
    loss_fake_inpaint = adv_loss(global_fake_out, 0)
    global_real_out = nets.discriminator(sample_org.x, sample_org.y)
    loss_real_inpaint = adv_loss(global_real_out, 1)



    loss = (loss_real + loss_fake) * args.lambda_adv_attr + \
        (loss_fake_inpaint + loss_real_inpaint) * args.lambda_adv_inpaint + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(), real_i=loss_real_inpaint.item(), real_f=loss_fake_inpaint.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())




def compute_g_loss(nets, args, masked_x, sample_org, sample_ref, masks=None, occ_mask=None):
    

    x_f, x_f_skips = nets.inpaint_encoder(x=masked_x, occ_mask=occ_mask)       # encode # for tilde
    # x_f : (B 512 16 16)


    # get src, target cond style maps
    s_map_src = nets.spatial_mapper(x_f, sample_org.y)
    s_map_target = nets.spatial_mapper(x_f, sample_ref.y)

    # decode
    x_rec = nets.inpaint_decoder(x=x_f, s=s_map_src, skip_connects=x_f_skips)    # G(x, source)
    x_fake = nets.inpaint_decoder(x=x_f, s=s_map_target, skip_connects=x_f_skips)    # G(x, target)

    
    # ----- adv loss
    out = nets.discriminator(x_fake, sample_ref.y)  # D(G(x, target), target)
    loss_adv = adv_loss(out, 1)

    # ---- inpaint adv loss
    x_fake = x_fake * occ_mask + sample_org.x * (1. - occ_mask)
    global_adv_out = nets.discriminator(x_fake, sample_ref.y) # improve
    loss_adv_inpaint = adv_loss(global_adv_out, 1)

    # ---- cycle loss
    loss_sty_src = 0
    x_rec = x_rec * occ_mask + sample_org.x * (1. - occ_mask) # improve
    zero_mask = torch.zeros_like(occ_mask).cuda()
    feature_gt_src, _ = nets.inpaint_encoder(x=sample_org.x, occ_mask=zero_mask)
    feature_rec_src, _ = nets.inpaint_encoder(x=x_rec, occ_mask=zero_mask)
    s_map_gt_src = nets.spatial_mapper(feature_gt_src, sample_org.y)
    s_map_rec_src = nets.spatial_mapper(feature_rec_src, sample_org.y)

    for i in range(len(s_map_gt_src)):
        loss_sty_src = loss_sty_src + torch.mean(torch.abs(s_map_gt_src[i] - s_map_rec_src[i]))   # from only 512x8x8
    loss_sty_src = loss_sty_src / len(s_map_gt_src)

    loss_sty_target = 0
    feature_gt_target, _ = nets.inpaint_encoder(x=sample_ref.x, occ_mask=zero_mask)
    feature_fake_target, _ = nets.inpaint_encoder(x=x_fake, occ_mask=zero_mask)
    s_map_gt_tgt = nets.spatial_mapper(feature_gt_target, sample_ref.y)
    s_map_rec_tgt = nets.spatial_mapper(feature_fake_target, sample_ref.y)
    
    
    for i in range(len(s_map_gt_tgt)):
        loss_sty_target = loss_sty_target + torch.mean(torch.abs(s_map_gt_tgt[i] - s_map_rec_tgt[i]))   # from only 512x8x8
    loss_sty_target = loss_sty_target / len(s_map_gt_tgt)
    
    loss_sty = loss_sty_src + loss_sty_target

    # ---- pixel loss
    loss_pix = nn.L1Loss()(x_rec, sample_org.x) # pixel (GT, G(x, source))

    # ---- percep loss
    loss_per = l1_norm(nets.resnet18(sample_org.x) - nets.resnet18(x_fake))    # percep(GT, G(x, target))

    # ---- identity loss
    loss_id = nn.MSELoss()(nets.recognizer(x_fake), nets.recognizer(sample_org.x))



    loss = loss_adv * args.lambda_adv_attr + loss_adv_inpaint * args.lambda_adv_inpaint \
           + args.lambda_sty * loss_sty \
           + args.lambda_per * loss_per \
           + args.lambda_pixel * loss_pix \
           + args.lambda_id * loss_id 
           

    return loss, Munch(adv=loss_adv.item(), adv_i=loss_adv_inpaint.item(),
                       sty=loss_sty.item(),
                       per=loss_per.item(),
                       pix=loss_pix.item(), id=loss_id.item())


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def l1_norm(x):
    return torch.mean(torch.abs(x))


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

def dis_forward(netD, ground_truth, x_inpaint):
    #assert ground_truth.size() == x_inpaint.size()
    batch_size = ground_truth.size()[0]
    batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
    batch_output = netD(batch_data)
    real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

    return real_pred, fake_pred