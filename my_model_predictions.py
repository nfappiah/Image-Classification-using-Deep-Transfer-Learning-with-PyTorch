# import all packages
from PIL import Image
import os
import numpy as np
import json
import torch
from torchvision import models
from torch import nn, optim

# define function to load model checkpoint
def load_checkpoint(filepath, device):
    # load checkpoint
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    #checkpoint = torch.load(filepath, map_location=device.type)
    
    # get model architecture
    arch = checkpoint['model_architecture']
    
    # define model
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define a custom classifier with the correct model architecture
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_layers'][0], checkpoint['output_size']),
        nn.LogSoftmax(dim=1)
    )
    
    # load model parameters
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # get learning rate
    learning_rate = checkpoint['learning_rate']
    
    # move model to device
    model.to(device)
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # get epochs
    epochs = checkpoint['epochs']
    
    # return epochs and optimizer as well to continue training
    return model, optimizer, epochs

# define function to process image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array. Input = path to single image, including the extension
    '''
    
    size = 256, 256
    
    # open image and manipulate it
    with Image.open(image_path) as im:
        # scale image
        im = im.resize(size) # resize
        
        # Determine the center coordinates
        width, height = im.size
        left = (width - 224) // 2
        top = (height - 224) // 2
        right = left + 224
        bottom = top + 224
        
        # Crop the center 224x224 portion
        im = im.crop((left, top, right, bottom))
        
        # convert from 0-255 to 0-1 using numpy
        np_im = np.array(im)
        np_im = np_im / 255.0
        
        # normalize each color chanel
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        np_im =  (np_im - means) / stds
        
        # re-order dimensions. PyTorch expects the color channel to be the first dimension 
        # but it's the third dimension in the PIL image and Numpy array.
        np_im = np_im.transpose((2,0,1))
        
        # convert numpy image array to torch tensor image array
        torch_im = torch.tensor(np_im)
        
    return torch_im

# define function to predict class probabilty and topk classes for a single image
def predict_topk(torch_im, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # convert torch_im to a float
    im = torch_im.to(torch.float)
    
    batch_size = 1 # because I am predicting only for 1 image
    im = im.unsqueeze(0).expand(batch_size, -1, -1, -1) # resize to a 4d array --> batch_size, channel, height, width
    
    # invert dictionary
    original_class_to_index = model.class_to_idx
    inverted_class_to_index = {v: k for k, v in original_class_to_index.items()}
    
    # move model to default device
    model.to(device)
    
    # predict
    model.eval()
    with torch.no_grad():
        im = im.to(device)
        logps = model.forward(im)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p[0].tolist() # convert to list
        top_class = top_class[0].tolist() # convert to list
        top_classes = [inverted_class_to_index[key] for key in top_class]
    
    return top_p, top_classes

# define a function to predict flower name
def predict_flower_name(probs, classes, category_names):
    # sort data before plotting
    sorted_data = sorted(zip(probs, classes), reverse=True)
    sorted_probs, sorted_classes = zip(*sorted_data)
    
    # load category names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # convert from class integer encoding to actual flower names
    flower_names = [cat_to_name[key] for key in sorted_classes]
    
    return flower_names, sorted_probs