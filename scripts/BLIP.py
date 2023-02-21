import torch

from models.blip import blip_decoder
from torchvision import transforms
from PIL import Image

def automatic_prompt(image,h,w):
    image_size = 512
        
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to('cuda')
    transform = transforms.Compose([
            transforms.Resize((h,w),interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image=transform(image.resize((image_size,image_size))).unsqueeze(0).to('cuda')  
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        return caption[0]