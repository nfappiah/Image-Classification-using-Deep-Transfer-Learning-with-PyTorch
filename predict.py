# import necessary packages
import argparse
import my_model_predictions
import torch
#print(torch.__version__)

# define main function
def main():
    # create an argument parser
    parser = argparse.ArgumentParser(description="Predict flower name and class probability of a single image.")
    
    # add arguments
    parser.add_argument("image_path", type=str, help="Path to image to be predicted.")
    parser.add_argument("filepath", type=str, help="Path to model checkpoint to be loaded.")
    parser.add_argument("--topk", type=int, default=1, help="The topk most likely classes.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Use a mapping of categories to real names:.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for computation")
    
    # parse the command-line arguments
    args = parser.parse_args()
   
    # Use GPU if it's available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type == 'cuda':
        print("Uing GPU for computation")
    else:
        print("Uing CPU for computation")
   
    # load model checkpoint
    [model, optimizer, num_epochs] = my_model_predictions.load_checkpoint(args.filepath, device)
    
    # process image
    torch_im = my_model_predictions.process_image(args.image_path)
    
    # predict class probabilty and topk classes for a single image
    [probs, classes] = my_model_predictions.predict_topk(torch_im, model, args.topk, device)
    
    # predict flower name
    [flower_names, sorted_probs] = my_model_predictions.predict_flower_name(probs, classes, args.category_names)
    
    print('flower name(s): {}'.format(flower_names))
    print('class probability: {}'.format(sorted_probs))

if __name__ == "__main__":
    main()
    