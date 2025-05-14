import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import clip
import torch
import torchray
import torch.nn as nn
from grad_cam import GradCAM
from torchray.attribution.grad_cam import grad_cam as tr_gradcam
import attention_utils_p as au


from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders, prepare_cub_dataloaders_extra
import torch.distributions as dist


from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16") 


device = "cuda"
# Path to the saved model checkpoint ---> this pth needs to be updated
checkpoint_path = '../ProbVLM/ckpt/ProbVLM_waterbirds_200epochs_12.12_best.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

class BayesCap_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        p_drop: dropout probability 
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        p_drop=0,
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_intr = self.mod(x)
        #print('dbg', x_intr.shape, x.shape)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta

class BayesCLIP(nn.Module):
    def __init__(
        self,
        model_path=None,
        device='cuda',
    ):
        super(BayesCLIP, self).__init__()
        self.clip_model = load_model_p(device, model_path)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.img_BayesCap = BayesCap_MLP(inp_dim=1024, out_dim=1024, hid_dim=512, num_layers=3, p_drop=0.3).to(device)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=1024, out_dim=1024, hid_dim=512, num_layers=3, p_drop=0.3).to(device)

    def forward(self, i_inputs, t_inputs):
        i_features, t_features = self.clip_model(i_inputs, t_inputs)

        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta), (i_features, t_features)


class BayesCap_for_CLIP(nn.Module):
    def __init__(
        self,
        inp_dim=1024,
        out_dim=1024,
        hid_dim=512,
        num_layers=3,
        p_drop=0.1,
    ):
        super(BayesCap_for_CLIP, self).__init__()
        self.img_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)

    def forward(self, i_features, t_features):
        
        # print('dbg', i_features.shape, t_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta)
    
    
def load_data_loader(dataset, data_dir, dataloader_config):
    prepare_loaders = {
        'coco': prepare_coco_dataloaders,
        'flickr': prepare_flickr_dataloaders,
        'CUB':prepare_cub_dataloaders,
        'FLO':prepare_flo_dataloaders
    }[dataset]
    if dataset == 'CUB':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root='../ProbVLM/Datasets/text_c10')
    elif dataset == 'FLO':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',)
    else:
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            vocab_path='ds/vocabs/coco_vocab.pkl')
    return loaders

def load_model_p(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='RN50',
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def sample_ggd(x_mu, x_1alpha, x_beta, num_samples=100):
    
    """
    
    Fucntion needs to be modified so that we can draw a sample from the distri
    Sample from a GGD with parameters (mu, alpha, beta).
    
    Args:
        x_mu: Tensor, the location parameter (mean).
        x_1alpha: Tensor, the scale parameter.
        x_beta: Tensor, the shape parameter.
        num_samples: int, number of samples to draw.
        
    Returns:
        feature_vector: Tensor, derived feature vector from GGD samples.
        
        
    """
    # Add a small epsilon to x_1alpha to avoid zero values
    epsilon = 1e-6
    x_1alpha_adjusted = x_1alpha + epsilon

    # Create an approximate normal distribution
    ggd_dist = dist.Normal(x_mu, x_1alpha_adjusted)

    # Sample and compute feature vector (e.g., mean of samples)
    samples = ggd_dist.sample((num_samples,))

    return samples

import pickle
import os

def save_data_loaders(loaders, filename):
    with open(filename, 'wb') as f:
        pickle.dump(loaders, f)

def load_data_loaders(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Usage
dataset = 'waterbird_1.0_forest2water2'  # coco or flickr
data_dir = ospj('../ProbVLM/Datasets/', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True
})

filename = '../ProbVLM/Datasets/CUB/data_loaders_waterbirds_12.12.pkl'
loaders = load_data_loaders(filename)

if loaders is None:
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    save_data_loaders(loaders, filename)


cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']


# In[14]:

device='cuda'
CLIP_Net = load_model_p(device=device, model_path=None)
ProbVLM_Net = BayesCap_for_CLIP(inp_dim=1024,
        out_dim=1024,
        hid_dim=512,
        num_layers=3,
        p_drop=0.1,
    )


# In[15]:

ProbVLM_Net = ProbVLM_Net.to(device)
ProbVLM_Net.load_state_dict(checkpoint)
ProbVLM_Net.eval()

text_list = ["an image of a bird", "a photo of a bird"]

class AttentionVLModel(nn.Module):
    def __init__(self, base_model, gradcam_layer='layer4.2.relu'):
        super(AttentionVLModel, self).__init__()
        self.base_model = base_model.module if hasattr(base_model, "module") else base_model
        self.gradcam = GradCAM(model=self.base_model, candidate_layers=[gradcam_layer])
    
    def forward(self, image_path, img_f, text_f, text_list, tok_t, tok_i, device):

        # Generate attention map
        attention_data = au.clip_gcam_prob(
            model=self.base_model,
            file_path=image_path,
            text_list=text_list,
            img_f = img_f,
            text_f = text_f,
            tokenized_text = tok_t,
            tokenized_img = tok_i,
            layer=self.gradcam.candidate_layers[0],
            device=device,
            plot_vis=False,
            save_vis_path = False
        )

        # Extract relevant outputs
        attentions, probs, unnorm_attentions, text = attention_data['attentions'], attention_data['probs'], attention_data['unnormalized_attentions'], attention_data['text_list']
        return attentions, probs, unnorm_attentions, text
    
attention_model = AttentionVLModel(base_model=CLIP_Net).to(device)

def token_to_text(coded_text):
    
    # Convert indices to tokens
    tokens = [tokenizer.convert_ids_to_tokens(indices.tolist()) for indices in coded_text]
    
    # Define a list of unwanted tokens
    unwanted_tokens = {'<|startoftext|>', '<|endoftext|>', '.', '!', '</w>'}

    # Filter the tokens to exclude unwanted ones and keep only the actual words
    filtered_words = [token[:-4] if token.endswith('</w>') else token for token in tokens if token not in unwanted_tokens]

    # Convert the list of words into a single string
    result_string = ' '.join(filtered_words)
    
    return [result_string]


# Function to process and save attention maps
def process_attention_maps(data_loader, save_folder, aggregation_method, max_files=15):
    # Define root and save paths
    ROOT = "../ProbVLM/Datasets/waterbird_1.0_forest2water2"
    SAVE_PATH = os.path.join(ROOT, save_folder)
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Starting to process data_loader in {save_folder} with {aggregation_method} aggregation method!")

    num_visualized = 0

    for i, batch in enumerate(data_loader):
        xI, xT, paths = batch[0].to(device), batch[1].to(device), batch[4]
        
        for t, (img, txt, path) in enumerate(zip(xI, xT, paths)):
            text_list = token_to_text(xT[t])

            # Filter by text content
            if text_list[0] not in {'a photo of a bird', 'an image of a bird'}:
                continue

            # Prepare inputs
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
            txt = txt.unsqueeze(0).to(device)

            with torch.no_grad():
                xfI, xfT = CLIP_Net(img, txt)
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = ProbVLM_Net(xfI, xfT)
                img_samples = sample_ggd(img_mu, img_1alpha, img_beta, 50)
                txt_samples = sample_ggd(txt_mu, txt_1alpha, txt_beta, 50)
            
            attentions_list = []
            unnormalized_attentions_list = []
            probss = []

            for img_feature_vector, txt_feature_vector in zip(img_samples, txt_samples):
                attentions, probs, unnorm_attentions, text_list = attention_model.forward(
                    image_path=path,
                    img_f=img_feature_vector.to(device),
                    text_f=txt_feature_vector.to(device),
                    text_list=text_list,
                    tok_t=txt,
                    tok_i=img,
                    device=device
                )
                attentions_list.append(attentions)
                unnormalized_attentions_list.append(unnorm_attentions)
                probss.append(probs)

            # Aggregate attentions
            if aggregation_method == 'median':
                aggregated_attention = torch.median(torch.stack(attentions_list), dim=0).values
                aggregated_unnorm_attention = torch.median(torch.stack(unnormalized_attentions_list), dim=0).values
                aggregated_probs = np.median(probss, axis=0)
            elif aggregation_method == 'mean':
                aggregated_attention = torch.mean(torch.stack(attentions_list), dim=0)
                aggregated_unnorm_attention = torch.mean(torch.stack(unnormalized_attentions_list), dim=0)
                aggregated_probs = np.mean(probss, axis=0)
            else:
                raise ValueError("Aggregation method must be 'mean' or 'median'")

            # Save paths and directories
            tail = path.split(ROOT)[-1]
            tail_path = os.path.join(SAVE_PATH, *tail.split(os.sep)[:-1])
            os.makedirs(tail_path, exist_ok=True)

            # Save .pth file
            attention_save_path = os.path.join(tail_path, f"{os.path.basename(path).replace('.jpg', '.pth')}")
            torch.save({
                'attentions': aggregated_attention,
                'unnormalized_attentions': aggregated_unnorm_attention,
                'probs': aggregated_probs,
                'text_list': text_list
            }, attention_save_path)

            # Save visualization
            os.makedirs(os.path.join(SAVE_PATH, 'vis'), exist_ok=True)
            save_vis_path = os.path.join(SAVE_PATH, 'vis', os.path.basename(path).replace('.jpg', '.jpg'))

            if num_visualized < max_files and i % 50 == 0:
                au.plot_attention_helper_p(
                    image=img,
                    attentions=[aggregated_attention],
                    unnormalized_attentions=[aggregated_unnorm_attention],
                    probs=[aggregated_probs],
                    text_list=text_list,
                    save_vis_path=save_vis_path,
                    resize=False
                )
                num_visualized += 1
    # Done statement after processing all batches in the data_loader
    print("Done!")

# Process and save for both aggregation methods
process_attention_maps(cub_train_loader, "clip_rn50_attention_gradcam_water_mean", "mean")
process_attention_maps(cub_valid_loader, "clip_rn50_attention_gradcam_water_mean", "mean")
process_attention_maps(cub_train_loader, "clip_rn50_attention_gradcam_water_median", "median")
process_attention_maps(cub_valid_loader, "clip_rn50_attention_gradcam_water_median", "median")
