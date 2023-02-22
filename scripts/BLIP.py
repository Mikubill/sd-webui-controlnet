import torch
from torchvision import transforms


def automatic_prompt(image,device):
    """
    If the user wants to use automatic_prompt, necessary load the model. 
    Therefore, it is needed to clone BLIP model.

    !pip install gitpython
    """
    import git
    git.Git("/").clone("https://github.com/salesforce/BLIP.git")
    from BLIP.models.blip import blip_decoder

    image_size = 512
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image=transform(image.resize((image_size,image_size))).unsqueeze(0).to(device)  
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    return caption[0]