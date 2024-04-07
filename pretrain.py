import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import numpy as np
from inspect import isfunction
import h5py
import argparse
from tqdm import tqdm 
import os
import logging
from scipy.stats import spearmanr, pearsonr
import timm
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from timm.models.vision_transformer import Block
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')

def compute_min_padding(height, width, target_height, target_width):
    pad_height = max(target_height - height, 0)
    pad_width = max(target_width - width, 0)

    return (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class IQADataset(Dataset):
    """
    IQA Dataset
    """
    def __init__(self, args, transform, status, loader=default_loader):
        """
        :param args:
        :param status: train/test
        :param loader: image loader
        """
        self.status = status
        self.loader = loader
        self.transform = transform
        self.num_avg_val = 15
        self.num_avg_train = 3
        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = args.K_fold
        k = args.k_test
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        train_index, test_index = [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)
            else:
                train_index.append(i)
        if 'train' in status:
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
        print('Index:')
        print(self.index)

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        self.label = []
        self.im_names = []
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))
            self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])
        width, height = im.size
        padding = compute_min_padding(height, width, 224, 224)
        new_transform = transforms.Compose([
                            transforms.Pad(padding, fill=0, padding_mode='constant'),
                            *self.transform.transforms
                        ])
        if self.status == 'train':
            ims = []
            for i in range(self.num_avg_train):
                im_ = new_transform(im)
                ims.append(im_)
            
            return ims, torch.Tensor([self.label[idx], ])
        elif self.status == 'test':
            ims = []
            for i in range(self.num_avg_val):
                im_ = new_transform(im)
                ims.append(im_)

            return ims, torch.Tensor([self.label[idx], ])

class IQARegression(nn.Module):
    def __init__(self, inchannels=768):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(inchannels, inchannels),
            nn.GELU(),
            nn.Linear(inchannels, 1)
        )
    
    def forward(self, x):

        pred = self.project(x)

        return pred

class Train:
    def __init__(self, args):
        self.opt = args
        self.create_model()
        self.init_data()
        self.fix_rate = 0.7
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam([
        {'params': self.vit.parameters()}, {'params': self.regressor.parameters()}], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.train()

    def create_model(self):
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
        self.regressor = IQARegression().to(device)

    def init_data(self):
        train_dataset = IQADataset(
            args,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(self.opt.crop_size),
                    transforms.ToTensor(),
                ]
            ),
            status = 'train',
        )
        test_dataset = IQADataset(
            args,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(self.opt.crop_size),
                    transforms.ToTensor(),
                ]
            ),
            status = 'test'
        )
        logging.info('number of train scenes: {}'.format(len(train_dataset)))
        logging.info('number of test scenes: {}'.format(len(test_dataset)))
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=True
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )

    def train_epoch(self, epoch):
        losses = []
        image_fix_num = "blocks.{}".format(int(12 * self.fix_rate))
        for name, parms in self.vit.named_parameters():
            parms.requires_grad_(False)
            if image_fix_num in name:
                break
        
        self.regressor.train()   

        pred_epoch = []
        labels_epoch = []
        
        for data in tqdm(self.train_loader):
            pred = 0
            for i in range(self.opt.num_avg_train):
                d_img_org = data[0][i].to(device)
                labels = data[1].to(device)

                vit_dis = self.vit.forward_features(d_img_org)[:, 0, :]

                pred += self.regressor(vit_dis)

            pred /= self.opt.num_avg_train

            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(pred.squeeze(-1)), labels.squeeze(-1))
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

        return ret_loss, rho_s, rho_p
    
    def eval_epoch(self, epoch):
        with torch.no_grad():
            losses = []
            self.vit.eval()
            self.regressor.eval()
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.test_loader):
                pred = 0
                for i in range(self.opt.num_avg_val):
                    d_img_org = data[0][i].to(device)
                    labels = data[1].to(device)

                    vit_dis = self.vit.forward_features(d_img_org)[:,0,:]

                    pred += self.regressor(vit_dis)
                    
                pred /= self.opt.num_avg_val
                # compute loss
                loss = self.criterion(torch.squeeze(pred.squeeze(-1)), labels.squeeze(-1))
                loss_val = loss.item()
                losses.append(loss_val)

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
            
            # compute correlation coefficient
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
            return np.mean(losses), rho_s, rho_p
          
    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'vit_model_state_dict': self.vit.state_dict(),
            'regressor_model_state_dict': self.regressor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))
    
    def train(self):
        main_score = 0
        for epoch in range(self.opt.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = self.train_epoch(epoch)
            if (epoch + 1) % self.opt.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))
                loss, rho_s, rho_p = self.eval_epoch(epoch)
                logging.info('Eval done...')

                if rho_s + rho_p > main_score:
                    main_score = rho_s + rho_p
                    print('Best now')
                    print('Best Main Score: {}'.format(main_score))
                    self.save_model( epoch, "best.pth", loss, rho_s, rho_p)
            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch VPMIQA")
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--K_fold', type=int, default=5,
                        help='K-fold cross-validation (default: 5)')
    parser.add_argument('--k_test', type=int, default=5,
                        help='The k-th fold used for test (1:K-fold, default: 5)')
    parser.add_argument('--crop_size', type=int, default=224, \
                        help='image size')
    parser.add_argument('--num_crop', type=int, default=1, \
                        help='random crop times')
    parser.add_argument('--num_workers', type=int, default=8, \
                        help='total workers')
    parser.add_argument('--batch_size', type=int, default=16, \
                        help='input batch size default=4')
    parser.add_argument('--learning_rate', type=float, default=1e-5, \
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, \
                        help='weight decay')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, \
                        help='drop path rate')   
    parser.add_argument('--T_max', type=int, default=50, \
                        help="cosine learning rate period (iteration)")
    parser.add_argument('--eta_min', type=int, default=0, \
                        help="mininum learning rate") 
    parser.add_argument('--n_epoch', type=int, default=200, \
                        help='total epoch for training')
    parser.add_argument('--patch_size', type=int, default=16, \
                        help='patch size of Vision Transformer')
    parser.add_argument('--val_freq', type=int, default=1, \
                        help='validation frequency')
    parser.add_argument('--num_avg_train', type=int, default=3, \
                    help='ensemble ways of train')
    parser.add_argument('--num_avg_val', type=int, default=15, \
                    help='ensemble ways of validation')
    parser.add_argument('--checkpoints_dir', type=str, default='/home/anonymous_dir/AIGC-IQA/checkpoint/', \
                        help='models are saved here')
    parser.add_argument('--database', default='KonIQ-10k', type=str, \
                        help='database name (default: LIVE)')
    parser.add_argument('--name', type=str, default='koniq10k', \
                        help='name of the experiment. It decides where to store samples and models')
    args = parser.parse_args()

    if args.database == 'KonIQ-10k':
        args.data_info = '/home/anonymous_dir/MoE-AGIQA/data/KonIQ-10k.mat'
        args.im_dir = '/home/anonymous_dir/1024x768/'

    return args

if __name__ == "__main__":
    setup_seed(20)
    currentDateAndTime = time.strftime("_%m%d_%H%M", time.localtime()) 
    print('------------------ ' + str(currentDateAndTime[1:]) + ' ------------------' )
    args = parse_args()
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.name + currentDateAndTime)
    os.makedirs(args.checkpoints_dir)
    Train(args)