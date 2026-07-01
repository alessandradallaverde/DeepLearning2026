from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import torch
from img2text_without_CLIP import main

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

net = main(clip_model, processor, exp_name="exp1", root=Path("/home/disi/deep_learning/DeepLearning2026/data"),)

# store the trained model
torch.save(net.state_dict(), "model_exp1.pth")