import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.optim as optim
from model.dispatcher import MODEL_DISPATCHER
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from utils.pytorchtools import EarlyStopping

DEVICE = 'cuda'

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
EPOCHS = 30
VALID_SIZE = 0.2

BASE_MODEL = 'resnet50'


def accuracy(outs, targets):
    top_p, top_class = outs.topk(1, dim=1)
    equals = top_class == targets.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    return accuracy
    

def train(dataloader, model, optimzer, criterion):

    model.train()
    running_loss = 0.0
    
    for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = data[0]
        target = data[1]
        image, target = image.to(DEVICE), target.to(DEVICE)

        optimzer.zero_grad()

        out = model(image)
        loss = criterion(out, target)
        loss.backward()
        optimzer.step()
        running_loss += loss.item()

    print("Train loss : {:.4f}".format(running_loss / len(dataloader)))


def evaluate(dataloader, model, criterion):
    
    model.eval()
    running_loss = 0.0
    acc = 0.0

    with torch.no_grad():

        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = data[0]
            target = data[1]
            image, target = image.to(DEVICE), target.to(DEVICE)

            out = model(image)
            loss = criterion(out, target)
            running_loss += loss.item()
            acc += accuracy(out, target)

    valid_loss = running_loss / len(dataloader)
    acc = acc / len(dataloader)
    print("Validation loss : {:.4f}\n Accuracy : {:.4f}".format(valid_loss, acc))

    return valid_loss


def main():

    #define train set transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    #define validation set transformations
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder('data/train_images/', transform=train_transform)
    valid_dataset = ImageFolder('data/train_images/', transform=valid_transforms)

    #split data : get indices of train and valid set
    valid_size = 0.2
    data_size = len(train_dataset)
    indices = list(range(data_size))

    #split indices
    train_indx, valid_indx, _, _ = train_test_split(indices, indices, test_size=valid_size, random_state=44)

    #create samplers from indices for train and validation sets. 
    train_sampler = SubsetRandomSampler(train_indx)
    valid_sampler = SubsetRandomSampler(valid_indx)

    #create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, sampler=valid_sampler)

    #create model
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    #model.load_state_dict(torch.load("model/checkpoints/checkpoint.pt"))
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5)
    early_stopping = EarlyStopping(patience=7, verbose=True)
    criterion = nn.CrossEntropyLoss()

    for e in range(EPOCHS):
        train(train_loader, model, optimizer, criterion)
        val_score = evaluate(valid_loader, model, criterion)
        scheduler.step(val_score)
        early_stopping(val_score, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break

if __name__ == '__main__':
    main()