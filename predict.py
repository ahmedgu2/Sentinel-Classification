import torch
from model.dispatcher import MODEL_DISPATCHER
from dataset.dataset import TestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

DEVICE = 'cuda'
BASE_MODEL = 'resnet34'
DATA_ROOT = 'data/test_images/'
BATCH_SIZE = 32

indx_2_class = {
    0:"AnnualCrop",
    1:"Forest",
    2:"HerbaceousVegetation",
    3:"highway",
    4:"Industrial",
    5:"Pasture",
    6:"PermanentCrop",
    7:"Residential",
    8:"River",
    9:"SeaLake"
}

def predict(dataloader, model):
    
    model.eval()
    acc = 0.0
    predictions = []

    with torch.no_grad():

        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = data['image']
            image_id = data['image_id']
            image = image.to(DEVICE)

            out = model(image)
            pred = np.argmax(out.cpu().detach().numpy(), axis=1)
            
            #save predictions
            for i in range(len(image_id)):
                predictions.append(("{}".format(image_id[i]), indx_2_class[pred[i]]))
            
    #save prediction to a file
    df = pd.DataFrame(predictions, columns=['image_id', 'class'])
    df.to_csv("predictions.csv", index=False)


def main():

    #define test set transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = TestDataset(DATA_ROOT, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    #create model
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=False)
    model.load_state_dict(torch.load('model/checkpoints/resnet34-0.0966.h5'))
    model.to(DEVICE)

    predict(test_loader, model)


if __name__ == '__main__':
    main()