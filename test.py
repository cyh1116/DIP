# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
# from model_baseline import DM2FNet, DM2FNet_woPhy
from model_brelu import DM2FNet, DM2FNet_woPhy
# from model_vggloss import DM2FNet,DM2FNet_woPhy,VGGLoss
# from model_vggloss_brelu import DM2FNet,DM2FNet_woPhy,VGGLoss
from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

torch.manual_seed(2018)
torch.cuda.set_device('cuda:3')

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
exp_name = 'O-Haze'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_18000_loss_0.05028_lr_0.000025', # brelu
    # 'snapshot': 'iter_14000_loss_0.51104_lr_0.000068' #vggloss
    
    # 'snapshot':'iter_20000_loss_0.05080_lr_0.000000', # baseline
    'snapshot':'iter_20000_loss_0.05047_lr_0.000000', # brelu
    # 'snapshot':'iter_20000_loss_0.51098_lr_0.000000' # vggloss
    # 'snapshot':'iter_20000_loss_0.51129_lr_0.000000' #vggloss+brelu
    
    # 'snapshot':'iter_18000_loss_0.05065_lr_0.000025' # baseline
    
    # 'snapshot': 'iter_16000_loss_0.05165_lr_0.000047' # baseline
    # 'snapshot':'iter_16000_loss_0.51154_lr_0.000047'  # vggloss
    # 'snapshot':'iter_16000_loss_0.05009_lr_0.000047' # brelu
     
    # 'snapshot':'iter_10000_loss_0.51478_lr_0.000107' #vggloss
    # 'snapshot':'iter_10000_loss_0.05251_lr_0.000107' #baseline
    # 'snapshot':'iter_10000_loss_0.05126_lr_0.000107' #brelu
    
    
    
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()
        # criterion = VGGLoss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'HazeRD_test')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims = [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,channel_axis=2,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}")


if __name__ == '__main__':
    print(torch.cuda.current_device())
    main()
