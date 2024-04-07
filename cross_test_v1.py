import torch
import h5py
import timm
import argparse
import ImageReward
import os
import scipy
import time
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import Block
from train_3k_224 import SaveOutput, IQARegression, setup_seed, get_vit_feature, compute_min_padding
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser(description='PyTorch source_code test on the test data')
    parser.add_argument("--model_file", type=str, default='/home/anonymous_dir/MoE-AGIQA/checkpoint/aigciqa2023_0329_1834/best.pth',
                        help="model file")
    parser.add_argument('--num_avg_val', type=int, default=15, \
                    help='ensemble ways of validation')
    args = parser.parse_args()

    args.data_info = '/home/anonymous_dir/MoE-AGIQA/data/AGIQA-1K.mat'

    Info = h5py.File(args.data_info, 'r')
    moss = Info['subjective_scores'][0, :]
    text_prompts = [Info[Info['text_prompt'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(moss)) ]
    files = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(moss)) ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose(
        [
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ]
    )
    prompt_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
    model_vit = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    model_regressor = IQARegression().to(device)
    reward = ImageReward.ImageReward(med_config='/home/anonymous_dir/models--THUDM--ImageReward/med_config.json', device=device).to(device)
    checkpoint = torch.load(args.model_file)

    reward.load_state_dict(checkpoint['reward_model_state_dict'],strict=False)
    model_regressor.load_state_dict(checkpoint['regressor_model_state_dict'])

    model_vit.eval()
    model_regressor.eval() 
    reward.eval()
    directory_name = "/home/anonymous_dir/AGIQA-1K/"

    with torch.no_grad():
        save_output = SaveOutput()
        hook_handles = []
        for layer in model_vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)
            preds = []
        with open('./output_2023_1k.txt', 'w') as file:
            for i in range(len(moss)):
                start_time = time.time()
                pred = 0
                filename = files[i]
                text_prompt = text_prompts[i]
                mos = moss[i]
                im = Image.open(os.path.join(directory_name, filename)).convert('RGB')
                width, height = im.size
                padding = compute_min_padding(height, width, 224, 224)
                new_transform = transforms.Compose([
                            transforms.Pad(padding, fill=0, padding_mode='constant'),
                            *transform.transforms
                        ])
                text_input = reward.blip.tokenizer(text_prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
                image_embeds = reward.blip.visual_encoder(prompt_transform(im).unsqueeze(0).to(device))
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
                text_output = reward.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
                text_features = text_output.last_hidden_state
                for j in range(args.num_avg_val):
                    im_ = new_transform(im).unsqueeze(0).to(device)
                    _ = model_vit(im_)
                    vit_dis = get_vit_feature(save_output)
                    save_output.outputs.clear()
                    B, N, C = vit_dis.shape
                    f_dis = vit_dis.transpose(1, 2).view(B, C, 14, 14)
                    pred += model_regressor(f_dis, text_features).item()
                pred /= args.num_avg_val
                preds.append(pred)
                end_time = time.time()
                print(f"Processing time for {filename}: {end_time - start_time:.2f} seconds")
                file.write(f"{filename}, {pred:.4f}, {mos}\n")
            srocc = scipy.stats.spearmanr(preds, moss)
            plcc = scipy.stats.pearsonr(preds, moss)
            print("SROCC:{:.4f}".format(srocc[0]))
            print("PLCC:{:.4f}".format(plcc[0]))
        for handle in hook_handles:
            handle.remove()