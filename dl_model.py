#This script is only used to pre-download the model in the container
import clip
import torch

print("Downloading CLIP model to container..")
global model, device
device = "cuda" if torch.cuda.is_available() else "cpu"
model,_ = clip.load("ViT-B/32", device=device)
