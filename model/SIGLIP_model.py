
import sys
import torch
from PIL import Image
import open_clip

class SIG_LIP():
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.model, _,  self.preprocess = open_clip.create_model_and_transforms('ViT-SO400M-16-SigLIP2-512', pretrained='webli',cache_dir="/root/train_caption/scripts/.cache")
        self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-16-SigLIP2-512')
        self.model.to(self.device).eval()

    def text_embedding(self, text):
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype('float32')[0]
        return text_features
    
    def image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(),torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image).to(self.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy().astype("float32")[0]
        return image_features
