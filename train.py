import os
from tqdm import tqdm
import torch
import torch.backends.cudnn
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
# import pytorch_ssim
# from math import log10

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

from loss import VGGLoss
from model import enhance_color
from data import LLIEDataset
from utils import weight_init# , logger, 
from config import get_config

# @logger
def load_data(cfg):
    train_data_transform = transforms.Compose([
        transforms.Resize([400, 600]),
        # transforms.CenterCrop([256, 256]),
        transforms.ToTensor()
    ])
    val_data_transform = transforms.Compose([
        # transforms.Resize([400, 600]),
        # transforms.CenterCrop([480, 480]),
        transforms.ToTensor()
    ])
    train_haze_dataset = LLIEDataset(cfg.ori_data_path, cfg.haze_data_path, train_data_transform, dataset_type=cfg.dataset_type, istrain=True)
    # train_haze_dataset.add_dataset(cfg.ori_data_path, cfg.ori_data_path, dataset_type=cfg.dataset_type,)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_haze_dataset = LLIEDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, val_data_transform, False, dataset_type=cfg.dataset_type)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)


# @logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.makedirs(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('enhance', epoch)))


# @logger
def load_network(device):
    net = enhance_color().to(device)
    net.apply(weight_init)
    return net


def load_pretrain_network(cfg, device):
    net = enhance_color().to(device)
    net.load_state_dict(torch.load(os.path.join(cfg.ckpt))['state_dict'])
    return net

# @logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


# @logger
def loss_func(device):
    criterion = torch.nn.L1Loss().to(device)
    return criterion


# @logger
def load_summaries(cfg):
    summary = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.net_name), comment='')
    return summary


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load summaries
    summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    vggloss = VGGLoss(device=device)
    # -------------------------------------------------------------------
    # load network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device)
    else:
        network = load_network(device)
    print('Total params: ', sum(p.numel() for p in network.parameters() if p.requires_grad))
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    # -------------------------------------------------------------------
    sample_dir = os.path.join(cfg.sample_output_folder, cfg.net_name)
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
        
    ssim = SSIM(data_range=1.).to(device=device)
    psnr = PSNR(data_range=1.).to(device=device)
    lpips = LPIPS(net_type='alex').to(device=device)
    
    # start train
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        
        total_loss = 0.


        train_bar = tqdm(train_loader)
        for step, (ori_image, LL_image, _) in enumerate(train_bar):
            count = epoch * train_number + (step + 1)
            ori_image, LL_image = ori_image.to(device), LL_image.to(device)           
            LLIE_image = network(LL_image,)
            recon_loss = criterion(LLIE_image, ori_image) 
            vgg_loss = vggloss(LLIE_image, ori_image)
            # ori_recon_loss = mseloss(rgb_to_lab(LLIE_image), rgb_to_lab(ori_image))
            # hep_loss = percept_loss(LLIE_image, ori_image)
            # hp_loss = hist_loss(LLIE_image, ori_image)
            # ssim_loss = 1 - ssim(LLIE_image, ori_image)
            loss =  recon_loss+ 1e-2 * vgg_loss  #recon_loss +1e-2 * hp_loss #+ 0.5 * ori_recon_loss
            total_loss = total_loss + loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            summary.add_scalar('loss', loss.item(), count)
            summary.add_scalar('recon_loss', recon_loss.item(), count)
            train_bar.set_description_str('Epoch: {}/{} | Step: {}/{} | lr: {:.6f} | Loss: {:.6f}-{:.6f}'
                  .format(epoch + 1, cfg.epochs, step + 1, train_number,
                          optimizer.param_groups[0]['lr'], 
                          total_loss/(step+1), recon_loss.item(), 
                        )
                    )
            
        scheduler.step()
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, cfg.epochs))
        
        network.eval()

        LLIE_valing_results = {'mse': 0, 'ssims': 0, 'psnrs': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'lpipss': 0, 'lpips': 0,} 
        
        val_bar = tqdm(val_loader)
        max_step = 20
        if len(val_loader.dataset) <max_step:
            max_step = len(val_loader.dataset)-1
        save_image = None
        for step, (ori_image, LL_image, _) in enumerate(val_bar):
            
            ori_image, LL_image = ori_image.to(device), LL_image.to(device)
            with torch.no_grad():
                LLIE_image = network(LL_image)
            
            LLIE_valing_results['batch_sizes'] += cfg.batch_size
            batch_psnr = psnr(LLIE_image, ori_image).item()
            batch_ssim = ssim(LLIE_image, ori_image).item()
            LLIE_valing_results['psnrs'] += batch_psnr * cfg.batch_size
            LLIE_valing_results['ssims'] += batch_ssim * cfg.batch_size
            
            LLIE_valing_results['psnr'] = LLIE_valing_results['psnrs'] / LLIE_valing_results['batch_sizes']
            LLIE_valing_results['ssim'] = LLIE_valing_results['ssims'] / LLIE_valing_results['batch_sizes']
            
            batch_lpips = lpips(LLIE_image, ori_image).item()
            LLIE_valing_results['lpipss'] += batch_lpips * cfg.batch_size
            LLIE_valing_results['lpips'] = LLIE_valing_results['lpipss'] / LLIE_valing_results['batch_sizes']

            if step <= max_step:   # only save image 10 times
                sv_im = torchvision.utils.make_grid(torch.cat((LL_image, LLIE_image, ori_image), 0), nrow=ori_image.shape[0])
                if save_image == None:
                    save_image = sv_im
                else:
                    save_image = torch.cat((save_image, sv_im), dim=2)
            if step == max_step:   # only save image 15 times
               torchvision.utils.save_image(
                    save_image,
                    os.path.join(sample_dir, '{}.jpg'.format(epoch + 1))
                )
            val_bar.set_description_str('[LLIE] PSNR: %.4f dB SSIM: %.4f LPIPS: %.4f' % (
                        LLIE_valing_results['psnr'], LLIE_valing_results['ssim'], LLIE_valing_results['lpips']))
            
        summary.add_scalar('Metrics/PSNR', LLIE_valing_results['psnr'], epoch)
        summary.add_scalar('Metrics/ssim', LLIE_valing_results['ssim'], epoch)
        summary.add_scalar('Metrics/lpips', LLIE_valing_results['lpips'], epoch)
        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch, cfg.model_dir, network, optimizer, cfg.net_name)
    # -------------------------------------------------------------------
    # train finish
    summary.close()


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
