import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BackgroundLoader(Dataset):
    def __init__(self, image_dirs, transform=None):
        self.image_paths = []
        for image_dir in image_dirs:
            self.image_paths.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((1024, 1024)), 
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image
