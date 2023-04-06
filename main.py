import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from model import BiGAN
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, train, transform=None):
        directory = "./datasets"
        
        imsize = 256
        self.images = []
        for file in os.listdir(directory):
            image = Image.open(directory + "/" + file).resize((imsize, imsize)).convert('RGBA')
            data = image.getdata()

            newData = []
            for item in data:
                if item[3] == 0:
                    newData.append((255, 255, 255))
                else:
                    newData.append(item[:3])

            image = image.convert('RGB')
            image.putdata(newData)
            self.images.append(image)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
            return image, 0

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='train', help="choose work type 'train' or 'test'")
    parser.add_argument('--epochs', default=401, type=int,help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int,help='mini-batch size (default: 32)')
    parser.add_argument('--early_stopping', default=50, type=int,
                        metavar='N', help='early stopping (default: 50)')
    
    # Model
    parser.add_argument('--encoder_lr', type=float, default=2e-4, help='learning rate for encoder')
    parser.add_argument('--generator_lr', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='learning rate for discriminator')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension of z')
    parser.add_argument('--weight_decay', type=float, default=2.5*1e-5, help='Weight decay')
    # Data
    parser.add_argument('--input_size', type=int, default=28, help='image size')
    parser.add_argument('--image_save_path', type=str, default='saved/generated_images', help='generated image save path')
    parser.add_argument('--model_save_path', type=str, default='saved/model_weight', help='model save path')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    return config


def main():
    config = parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5,])])
    # MNIST dataset 
    train_dataset = CustomImageDataset(train=True, transform=transform)
    test_dataset = CustomImageDataset(train=False, transform=transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False)

    # Model
    model = BiGAN(config)
    model.train(train_loader)

if __name__ == '__main__':
    main()