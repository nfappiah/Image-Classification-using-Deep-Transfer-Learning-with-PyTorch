# import all packages
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

# define function to load and transform data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data

# define function to build and train network
def ffwd_model(arch, learning_rate, hidden_units, epochs, device, trainloader, validloader):
    # check if the user has provided valid inputs for arch
    assert arch == "vgg11" or arch == "vgg13" or arch == "vgg16" or arch == "vgg19" or arch == "alexnet", \
        "Please use vgg11 or vgg13 or vgg16 or vgg19 or alexnet. The default is alexnet if arch is not specified."
    
    # define model
    model = models.__dict__[arch](pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # define classifier
    if arch == 'vgg11' or arch == 'vgg13' or arch == 'vgg16' or arch == 'vgg19':
        print('Using one of the vgg\'s')
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    elif arch == 'alexnet':
        print('Using alexnet')
        model.classifier = nn.Sequential(nn.Linear(9216, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    # define criterion
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # move model to default device
    model.to(device);
    
    # Train network
    steps = 0
    running_loss = 0
    print_every = 5 # print very 5 batches. 41 batches make one epoch with a batch size of 32
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader: # 26 batches with a batch size of 32
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    return model, optimizer

# define function to do validation on test set
def validate_model(model, testloader, device):
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader: # 26 batches with a batch size of 32
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

# define function to save checkpoint
def save_checkpoint(model, save_dir, hidden_units, train_data, arch, optimizer, epochs, learning_rate):
    # save class to index
    model.class_to_idx = train_data.class_to_idx # need to save this with the checkpoint
    
    if arch == 'vgg11' or arch == 'vgg13' or arch == 'vgg16' or arch == 'vgg19':
        checkpoint = {'model_architecture': arch,
                      'learning_rate': learning_rate,
                      'input_size': 25088,
                      'output_size': 102,
                      'hidden_layers': [hidden_units, 102],
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epochs': epochs
                     }
    elif arch == 'alexnet':
        checkpoint = {'model_architecture': arch,
                      'learning_rate': learning_rate,
                      'input_size': 9216,
                      'output_size': 102,
                      'hidden_layers': [hidden_units, 102],
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epochs': epochs
                     }

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Specify the complete file path including the directory
    file_path = os.path.join(save_dir, 'checkpoint.pth')

    # save the checkpoint with the specified file path
    torch.save(checkpoint, file_path)
    
    return file_path
    