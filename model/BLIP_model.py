from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

class BLIP():
    def __init__(self,ckpt_path=None) -> None:
        self.device = "cuda"
        if not ckpt_path:
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl-coco")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl-coco").to(self.device)
        else :
            self.processor = AutoProcessor.from_pretrained(ckpt_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(ckpt_path,local_files_only=True).to(self.device)
        self.model.eval()

    def raw_image_captioning(self, img_path: str) -> str:
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def prompted_image_captioning(self, img_path: str, text="Describe the image carefully. Focus on the main subject, features and characteristics.") -> str:
        image = Image.open(img_path)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text