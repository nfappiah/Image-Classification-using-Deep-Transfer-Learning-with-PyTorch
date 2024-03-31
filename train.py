# import necessary packages
import argparse
import my_model
import torch
import os

# define main function
def main():
    # get current working directory to save checkpoint
    current_directory = os.getcwd()
    
    # create an argument parser
    parser = argparse.ArgumentParser(description="Train a neural network.")
    
    # add arguments
    parser.add_argument("data_dir", type=str, help="data directory where the data are located.")
    parser.add_argument("--arch", type=str, default="alexnet", help="The selected pre-trained model.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The selected learning rate.")
    parser.add_argument("--hidden_units", type=int, default=200, help="The selected number of hidden units.")
    parser.add_argument("--epochs", type=int, default=10, help="The selected number of epochs.")
    parser.add_argument("--save_dir", type=str, default=current_directory, help="The directory where the checkpoint is saved.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for computation")
    
    # parse the command-line arguments
    args = parser.parse_args()
   
    # Use GPU if it's available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type == "cuda":
        print("Uing GPU for computation")
    else:
        print("Uing CPU for computation")
    
    # load and transform data
    [trainloader, validloader, testloader, train_data] = my_model.load_data(args.data_dir)
    
    # build and train neural network. optimizer and epochs are required if one wants to continue training
    [model, optimizer] = my_model.ffwd_model(args.arch, args.learning_rate, args.hidden_units, args.epochs, device, trainloader, validloader)
    
    # validate model on test set
    print('End of training. Validating model on test set.')
    my_model.validate_model(model, testloader, device)
    
    # save model checkpoint
    print('Saving model checkpoint.')
    file_path = my_model.save_checkpoint(model, args.save_dir, args.hidden_units, train_data, args.arch, optimizer, args.epochs, args.learning_rate)

if __name__ == "__main__":
    main()

    