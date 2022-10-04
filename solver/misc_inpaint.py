import torch
import os
import copy
from utils.image import save_image
from data.loader import get_eval_loader
from tqdm import tqdm
from utils.file import make_path

from solver.mask_tools import mask_image_predefined, load_predefined_mask, load_predefined_mask_noshuffle


# code is based on https://github.com/songquanpeng/L2M-GAN
@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, filename):
    x_concat = [x_src]
    for y_trg in y_trg_list:
        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            x_fake, x_fake_inp = nets.generator(x_src, s_trg)
            #x_fake = nets.generator(x_src, s_trg)
            x_concat += [x_fake]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, x_src.size()[0], filename)


@torch.no_grad()
def translate_using_label(nets, args, sample_src, y_trg_list, filename):
    #x_concat = [sample_src.x]
    x_concat = []

    # -------- masking parts ------------
    # [1] ------- MASKING ------------
    occ_mask = load_predefined_mask(args, batch_size=sample_src.x.size(0))
    sample_masked, occ_mask = mask_image_predefined(args, sample_src.x, occ_mask)
    
    # from upper, gt, input, coarse, ---
    x_concat += [sample_src.x]
    x_concat += [sample_masked]

    for y_trg in y_trg_list:

        x_f, x_f_skips = nets.inpaint_encoder(x=sample_masked, occ_mask=occ_mask)       # encode # for tilde
        s_map_target = nets.spatial_mapper(x_f, y_trg)
        x_fake = nets.inpaint_decoder(x=x_f, s=s_map_target, skip_connects=x_f_skips)    # decode

        x_fake = x_fake * occ_mask + sample_src.x * (1. - occ_mask)
        x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, sample_src.x.size()[0], filename)


@torch.no_grad()
def generate_samples(nets, args, path):
    args = copy.deepcopy(args)
    args.batch_size = args.eval_batch_size
    for src_idx, src_domain in enumerate(args.domains):
        loader = get_eval_loader(path=os.path.join(args.test_path, src_domain), **args)
        N = args.eval_batch_size
        target_domains = [domain for domain in args.domains]
        src_class_label = torch.tensor([src_idx] * N).to(args.device)
        for trg_idx, trg_domain in enumerate(target_domains):
            
            
            if trg_domain == src_domain:       # if need all, comment this
                continue
            trg_class_label = torch.tensor([trg_idx] * N).to(args.device)
            save_path = os.path.join(path, f"{src_domain}2{trg_domain}")
            make_path(save_path)

            # our temp
            if os.path.exists(os.path.join(save_path, 'masked')) == False:
                print("generated " + os.path.join(save_path, 'masked'))
                os.makedirs(os.path.join(save_path, 'masked'))

            # mask_idx for not shuffling
            chosen_mask_idx = 0

            for i, (query_image, image_name) in enumerate(tqdm(loader, total=len(loader))):
                query_image = query_image.to(args.device)
                
                images = []

                # -------- masking parts ------------
                # [1] ------- MASKING ------------
                if args.mask_not_shuffle:
                    occ_mask,  chosen_mask_idx = load_predefined_mask_noshuffle(args, batch_size=query_image.size(0), chose_idx=chosen_mask_idx)
                else:
                    occ_mask = load_predefined_mask(args, batch_size=query_image.size(0))
                sample_masked, occ_mask = mask_image_predefined(args, query_image, occ_mask)


                for j in range(args.eval_repeat_num):
                    x_f, x_f_skips = nets.inpaint_encoder(x=sample_masked, occ_mask=occ_mask)       # encode # for tilde
                    s_map_target = nets.spatial_mapper(x_f, trg_class_label)
                    x_fake = nets.inpaint_decoder(x=x_f, s=s_map_target, skip_connects=x_f_skips)    # decode

                    x_fake = x_fake * occ_mask + query_image * (1. - occ_mask)

                    images.append(x_fake)
                    for k in range(N):
                        filename = os.path.join(save_path, image_name[k])
                        save_image(x_fake[k], col_num=1, filename=filename)

                        # ------------ our temp code for saving masked image....
                        #maskdir = os.makedirs(os.path.join(save_path, 'masked'))
                        #maskedname = os.path.join(save_path, 'masked', image_name[k])
                        #save_image(sample_masked[k], col_num=1, filename=maskedname)