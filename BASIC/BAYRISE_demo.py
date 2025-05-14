# BayesVLM + RISE Attribution Demo (CLIP ViT-B/32)

import sys
import os

# Add BayesVLM to Python path
sys.path.append(os.path.abspath("../BayesVLM"))

from bayesvlm_model import BayesCLIP


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import cv2

# Load BayesVLM wrapper
torch.set_grad_enabled(False)
from bayesvlm.bayesvlm_model import BayesCLIP

# ---- Load Model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BayesCLIP('ViT-B/32', device=device)
model.eval()

# ---- Load Image and Text Prompt ----
image_path = "test.jpg"  # <-- Replace with your image path
text_prompt = "a photo of a dog"

preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

img = Image.open(image_path).convert("RGB")
image_tensor = preprocess(img).unsqueeze(0).to(device)

# Tokenize text
import clip  # Just for tokenizer
text = clip.tokenize([text_prompt]).to(device)

# ---- Get Text Embedding Distribution ----
text_mean, text_cov = model.encode_text(text, return_distribution=True)
text_mean = text_mean[0]
text_cov = text_cov[0]

# ---- Generate Random Masks ----
def generate_masks(N=1000, s=7, p=0.5, image_size=224):
    masks = np.random.binomial(1, p, size=(N, s, s)).astype('float32')
    masks = np.stack([
        cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        for m in masks
    ])
    masks = masks[..., np.newaxis]  # [N, H, W, 1]
    return masks

N_MASKS = 1000
masks = generate_masks(N=N_MASKS)

# ---- Cosine Expectation Function ----
def expected_cosine_similarity(mu1, cov1, mu2, cov2):
    dot = mu1 @ mu2
    norm1_sq = mu1 @ mu1 + torch.trace(cov1)
    norm2_sq = mu2 @ mu2 + torch.trace(cov2)
    denom = (norm1_sq.sqrt() * norm2_sq.sqrt()) + 1e-6
    return (dot / denom).item()

# ---- Apply RISE ----
saliency_map = np.zeros((224, 224))

for i in range(N_MASKS):
    m = masks[i]  # [H, W, 1]
    m_tensor = torch.tensor(m.transpose(2, 0, 1)).to(device)  # [1, H, W]
    masked_img = image_tensor * m_tensor

    im_mean, im_cov = model.encode_image(masked_img, return_distribution=True)
    im_mean = im_mean[0]
    im_cov = im_cov[0]

    score = expected_cosine_similarity(im_mean, im_cov, text_mean, text_cov)
    saliency_map += m[..., 0] * score

# Normalize
saliency_map /= np.sum(masks, axis=0)[..., 0] + 1e-8

# ---- Visualize ----
original_img = np.array(img.resize((224, 224))) / 255.0
plt.imshow(original_img)
plt.imshow(saliency_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title("RISE + BayesVLM Attribution")
plt.show()
