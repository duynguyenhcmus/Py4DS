# =========================================================================== #
'''
                            Lab07: Using Pytorch
                Project: Transfer Learning for Computer Vision 
    - Purpose: Learn how to train a convolutional neural network for image 
    classification using transfer learning.
    - Problem: Train a model to classify ants and bees.
    - Input: 
        + About 120 training images each for ants and bees. 
        + 75 validation images for each class.
    - Output: classified images. 
'''
# =========================================================================== #
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def imshow(inp, title=None):    
    '''
        Purpose: Show a few images.
        Input: 
            - inp : tourch.Tensor

            - title : string/list of strings
                label's names of images
        Output: 
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, dataloaders, device,\ 
                dataset_sizes, num_epochs=25):
    '''
        Purpose: 
        Input: 
            - model : 
            - criterion :
            - optimizer :
            - scheduler :
            - dataloaders :
            - device :
            - dataset_sizes :
            - num_epochs : int (default = 25)
                
        Output: 
    '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, device, class_names, num_images=6):
    '''
        Purpose: 
        Input: 
            - model : tourch.Tensor
            - dataloaders :
            - device : 
            - class_names :
            - num_images : int (default = 6)
                
        Output: 
    '''
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    plt.ion()   # interactive mode
    # ======================================================================= #
    ''' 
        LOADING DATA
        We use torchvision and torch.utils.data packages for loading the data.
    '''
    # ======================================================================= #
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),\
                                            data_transforms[x])\
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,\
                                                shuffle=True, num_workers=4)\
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ======================================================================= #
    ''' 
        VISUALIZING A FEW TRAINING IMAGES
    '''
    # ======================================================================= #
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    print(type(out))
    plt.figure(figsize = (10,10))
    imshow(out, title=[class_names[x] for x in classes])
    
    # ======================================================================= #
    '''
        FINETUNING THE CONVNET
        Load a pretrained model and reset final fully conneted layer.
    '''
    # ======================================================================= #
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # ======================================================================= #
    '''
        RAINING AND EVALUATING
    '''
    # ======================================================================= #
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,\
                           dataloaders, device, dataset_sizes, num_epochs=25)

    # Visualize the model predictions
    visualize_model(model_ft, dataloaders, device, class_names)

    # ======================================================================= #
    '''
        CONVNET AS FIXED FEATURE EXTRACTOR
    '''
    # ======================================================================= #
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # ======================================================================= #
    '''
        TRAINING AND EVALUATING
    '''
    # ======================================================================= #
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,\
                             dataloaders, device, dataset_sizes, num_epochs=25)

    visualize_model(model_conv, dataloaders, device, class_names)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()