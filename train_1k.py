import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import ImageReward
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
    
def get_vit_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[4][:, 1:, :],
            save_output.outputs[5][:, 1:, :],
            save_output.outputs[6][:, 1:, :],
            save_output.outputs[7][:, 1:, :]
        ),
        dim=2
    )
    return feat


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

class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []

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
        self.prompt_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
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
        self.txt_prompts = [Info[Info['text_prompt'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        self.label = []
        self.im_names = []
        self.text_prompt = []
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))
            self.label.append(self.mos[idx])
            self.text_prompt.append(self.txt_prompts[idx])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])
        prompt_im = self.prompt_transform(im)
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
            
            return prompt_im, ims, torch.Tensor([self.label[idx], ]), self.text_prompt[idx]
        elif self.status == 'test':
            ims = []
            for i in range(self.num_avg_val):
                im_ = new_transform(im)
                ims.append(im_)

            return prompt_im, ims, torch.Tensor([self.label[idx], ]), self.text_prompt[idx]  
            
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1)
        )
    def forward(self, x):
        return self.project(x.unsqueeze(0)).squeeze(0)

class IQARegression(nn.Module):
    def __init__(self, inchannels=768, outchannels=512):
        super().__init__()

        self.down_channel= nn.Conv2d(inchannels*4 , inchannels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1), 
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        scale = inchannels ** -0.5
        self.cross_attention = CrossAttention(outchannels)
        self.norm1 = nn.LayerNorm(outchannels)
        self.norm2 = nn.LayerNorm(outchannels)
        self.proj = nn.Parameter(scale * torch.randn(inchannels, outchannels))

        self.gating_network = nn.Sequential(
            nn.Linear(outchannels, 4),
            nn.Softmax(dim=1)
        )
        proj_experts = []
        self.proj_nums = 4
        for i in range(self.proj_nums):
            proj_experts.append(Projection(outchannels, outchannels))
        self.proj_experts = nn.Sequential(*proj_experts)
    
    def forward(self, x, text_features):

        f_dis = self.down_channel(x)

        f_dis = self.conv(f_dis)

        B, C, W, H = f_dis.shape
        L = W*H

        f_dis = f_dis.view(B, C, L).permute(0, 2, 1).contiguous()
        
        text_features = text_features @ self.proj

        f_dis = self.norm1(f_dis)

        f_dis = f_dis + self.cross_attention(f_dis, self.norm2(text_features))

        f_dis = f_dis.permute(0, 2, 1).contiguous().view(B, C, W, H)

        f_dis = self.pool(f_dis)

        f_dis = f_dis.view(f_dis.size(0), -1)

        gating_weight = self.gating_network(f_dis)

        gating_weight_value, gating_weight_index = torch.topk(gating_weight, k=3, dim=1)

        preds = torch.tensor([]).cuda()
                                        
        for i in range(f_dis.size(0)):
            preds_one = torch.tensor([]).cuda()
            for j in gating_weight_index[i]:
                _pred = self.proj_experts[j](f_dis[i])
                preds_one = torch.cat((preds_one, _pred.unsqueeze(0)), 0)
            preds = torch.cat((preds, preds_one.unsqueeze(0)), 0)

        pred = torch.sum(preds * gating_weight_value.unsqueeze(2), dim=1)

        return pred

class Train:
    def __init__(self, args):
        self.opt = args
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.fix_rate = 0.5
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam([
        {'params': self.reward.parameters()}, {'params': self.regressor.parameters()}], lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.train()

    def create_model(self):
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
        self.regressor = IQARegression().to(device)
        self.reward = ImageReward.load("ImageReward-v1.0").to(device)

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

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
        self.vit.eval()
        self.reward.train()
        for name, parms in self.reward.mlp.named_parameters():
            parms.requires_grad_(False)
        for name, parms in self.reward.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        if self.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * self.fix_rate))
            image_fix_num = "blocks.{}".format(int(24 * self.fix_rate))
            for name, parms in self.reward.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.reward.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break
        
        self.regressor.train()   

        pred_epoch = []
        labels_epoch = []
        
        for data in tqdm(self.train_loader):
            pred = 0
            image_prompt = data[0]
            text_prompt = data[3]
            text_input = self.reward.blip.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
            image_embeds = self.reward.blip.visual_encoder(image_prompt.to(device))
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
            text_output = self.reward.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
            text_features = text_output.last_hidden_state
            for i in range(self.opt.num_avg_train):
                d_img_org = data[1][i].to(device)
                labels = data[2].to(device)

                _ = self.vit(d_img_org)
                vit_dis = get_vit_feature(self.save_output)

                self.save_output.outputs.clear()

                B, N, C = vit_dis.shape
                if self.opt.patch_size == 16:
                    H,W = 14, 14
                else:
                    H,W = 28,28
                assert H*W==N 
                f_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

                pred += self.regressor(f_dis, text_features)

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
            self.reward.eval()
            self.regressor.eval()
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.test_loader):
                pred = 0
                image_prompt = data[0]
                text_prompt = data[3]
                text_input = self.reward.blip.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
                image_embeds = self.reward.blip.visual_encoder(image_prompt.to(device))
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
                text_output = self.reward.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
                text_features = text_output.last_hidden_state
                for i in range(self.opt.num_avg_val):
                    d_img_org = data[1][i].to(device)
                    labels = data[2].to(device)

                    _ = self.vit(d_img_org)
                    vit_dis = get_vit_feature(self.save_output)

                    self.save_output.outputs.clear()

                    B, N, C = vit_dis.shape
                    if self.opt.patch_size == 16:
                        H,W = 14, 14
                    else:
                        H,W = 28,28
                    assert H*W==N 
                    f_dis = vit_dis.transpose(1, 2).view(B, C, H, W)
                    pred += self.regressor(f_dis, text_features)
                    
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
            'reward_model_state_dict': self.reward.state_dict(),
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
    parser.add_argument('--n_epoch', type=int, default=100, \
                        help='total epoch for training')
    parser.add_argument('--patch_size', type=int, default=16, \
                        help='patch size of Vision Transformer')
    parser.add_argument('--val_freq', type=int, default=1, \
                        help='validation frequency')
    parser.add_argument('--num_avg_train', type=int, default=3, \
                    help='ensemble ways of train')
    parser.add_argument('--num_avg_val', type=int, default=15, \
                    help='ensemble ways of validation')
    parser.add_argument('--checkpoints_dir', type=str, default='/home/anonymous_dir/MoE-AGIQA/checkpoint/', \
                        help='models are saved here')
    parser.add_argument('--database', default='AGIQA-1K', type=str, \
                        help='database name (default: AGIQA-1K)')
    parser.add_argument('--name', type=str, default='agiqa1K', \
                        help='name of the experiment. It decides where to store samples and models')
    args = parser.parse_args()

    if args.database == 'AGIQA-1K':
        args.data_info = '/home/anonymous_dir/MoE-AGIQA/data/AGIQA-1K.mat'
        args.im_dir = '/home/anonymous_dir/AGIQA-1K/'

    return args

if __name__ == "__main__":
    setup_seed(20)
    currentDateAndTime = time.strftime("_%m%d_%H%M", time.localtime()) 
    print('------------------ ' + str(currentDateAndTime[1:]) + ' ------------------' )
    args = parse_args()
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.name + currentDateAndTime)
    os.makedirs(args.checkpoints_dir)
    Train(args)