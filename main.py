import time
print(time.time())
import numpy as np
import pandas as pd
import timm
import random
import os
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import tqdm
from torch import nn, optim

from utils.engine import train_one_epoch, validate
from datasets.cassava_dataset import CassavaDataset

import json

print(time.time())

SEED = 123456789

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

hyper_params = {
    'EXP_NAME': 'exp_07',

    'MODEL_NAME': 'tf_efficientnet_b3_ns',
    'IM_SIZE': 300,

    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-3,
    'LR_STEP': 4,
    'EPOCH': 16,
    'WARMUP_EPOCH': 16,
    'WEIGHT_DECAY': 1e-4,
    'GRADIENT_COEFF': 2,
    #'NORM_WEIGHTS': [0.4070720457073391, 0.20214130364727165, 0.18545151453641143, 0.03362876680984022, 0.1717063692991376],
    'NORM_WEIGHTS': None,

    'VAL_BATCH_SIZE': 32,

    'NUM_WORKER': 16,
}

data = pd.read_csv('cassava-leaf-disease-classification/train.csv')
choosen_prob = np.random.rand(len(data))
train_df = data[choosen_prob >= 0.2]
val_df = data[choosen_prob < 0.2]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COLOR_JILTER = 0.3

train_transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.ColorJitter(brightness=COLOR_JILTER, contrast=COLOR_JILTER, saturation=COLOR_JILTER, hue=COLOR_JILTER),
     transforms.Normalize(MEAN, STD),
     transforms.Resize((hyper_params['IM_SIZE'], hyper_params['IM_SIZE']), interpolation=Image.BICUBIC),
     transforms.RandomHorizontalFlip(),
     #transforms.RandomRotation(5),
     #transforms.RandomResizedCrop((hyper_params['IM_SIZE'], hyper_params['IM_SIZE']))
     ])

val_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((hyper_params['IM_SIZE'], hyper_params['IM_SIZE']), interpolation=Image.BICUBIC),
     transforms.Normalize(MEAN, STD)])
    
train_image_dir = 'cassava-leaf-disease-classification/train_images'
train_dataset = CassavaDataset(train_image_dir, train_df, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=hyper_params['BATCH_SIZE'], 
                                           shuffle=True, 
                                           num_workers=hyper_params['NUM_WORKER'])

val_image_dir = 'cassava-leaf-disease-classification/train_images'
val_dataset = CassavaDataset(val_image_dir, val_df, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                           batch_size=hyper_params['VAL_BATCH_SIZE'], 
                                           shuffle=False, 
                                           num_workers=hyper_params['NUM_WORKER'])

model = timm.create_model(hyper_params['MODEL_NAME'], pretrained=True, num_classes=5)
model.eval()
model.classifier.train()
model.conv_head.train()
model.bn2.train()
model.cuda()

class_weights = None
if hyper_params['NORM_WEIGHTS'] != None:
    class_weights = torch.FloatTensor(hyper_params['NORM_WEIGHTS']).cuda()
    
criterion = nn.CrossEntropyLoss(weight=class_weights)
linear_scaled_lr = 8.0 * hyper_params['LEARNING_RATE'] * hyper_params['GRADIENT_COEFF'] * hyper_params['BATCH_SIZE'] / 512.0
optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9, weight_decay=hyper_params['WEIGHT_DECAY'])
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyper_params['LR_STEP'])

if __name__ == '__main__':
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    with open('logs/logs_{}_{}.txt'.format(hyper_params['EXP_NAME'], int(time.time())), 'w') as logs_file:
        logs_file.write(json.dumps(hyper_params))
        logs_file.write('\n')
        logs_file.write('**************************************************\n')

        for epoch in range(hyper_params['EPOCH']):
            train_loss = 0.0
            best_acc = 0
            train_loss, train_acc = train_one_epoch(epoch,
                                                    model,
                                                    train_loader,
                                                    criterion,
                                                    optimizer,
                                                    lr_scheduler,
                                                    gradient_to_accumulation=hyper_params['GRADIENT_COEFF'])

            logs_file.write('Training loss at epoch {}: {:.4f}\n'.format(epoch, train_loss))
            logs_file.write('Training acc at epoch {}: {:.4f}\n'.format(epoch, train_acc))

            val_loss, val_acc, current_best_acc = validate(epoch,
                                                           model,
                                                           val_loader,
                                                           criterion,
                                                           exp_name=hyper_params['EXP_NAME'],
                                                           current_best_acc=best_acc)
            best_acc = current_best_acc

            logs_file.write('Validation loss at epoch {}: {:.4f}\n'.format(epoch, val_loss))
            logs_file.write('Validation acc at epoch {}: {:.4f}\n'.format(epoch, val_acc))

            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
