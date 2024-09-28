import argparse
import torch
from torchvision import models
from PIL import Image
import json
import numpy as np

def get_input_arguments():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    
    parser.add_argument('input', type=str, help='Path to the image file you want to classify.')
    parser.add_argument('checkpoint', type=str, help='Path to the file containing the trained model.')
    
    # Add optional arguments (the user can choose to provide these or not)
    parser.add_argument('--top_k', type=int, default=5, 
                        help='How many top predictions to show. Default is 5.')
    parser.add_argument('--category_names', type=str, 
                        help='Path to a file that gives flower names for each category.')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use this flag if you want to use a GPU for faster prediction.')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    saved_checkpoint = torch.load(filepath)
    
    if saved_checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
    elif saved_checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(weights='IMAGENET1K_V1')
    else:
        print(f"Sorry, we don't support the {saved_checkpoint['architecture']} architecture.")
        exit()
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = saved_checkpoint['classifier']
    
    model.load_state_dict(saved_checkpoint['state_dict'])
    
    model.class_to_idx = saved_checkpoint['class_to_idx']
    
    return model
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    width, height = image.size
    if width < height:
        image.thumbnail((256, 256 * height // width))
    else:
        image.thumbnail((256 * width // height, 256))
    
    # Center crop
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(image) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
def predict_image(image_path, model, topk, device):
    # Process image
    np_image = process_image(image_path)
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model.forward(tensor_image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        
        # Convert to lists
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        
        # Invert class_to_idx
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[cls] for cls in top_class]
    
    return top_p, top_classes
def main():
    args = get_input_arguments()
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU for predictions")
    else:
        device = torch.device('cpu')
        print("Using CPU for predictions")
    
    print(f"Loading model from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    print(f"Predicting top {args.top_k} classes for image: {args.input}")
    top_probabilities, top_classes = predict_image(args.input, model, args.top_k, device)
    
    if args.category_names:
        print(f"Loading category names from {args.category_names}")
        with open(args.category_names, 'r') as f:
            category_to_name = json.load(f)
        top_flowers = [category_to_name[str(cls)] for cls in top_classes]
    else:
        top_flowers = top_classes
    
    print("\nTop K Classes and Probabilities:")
    for i, (probability, flower) in enumerate(zip(top_probabilities, top_flowers), 1):
        print(f"{i}. {flower}: {probability:.3f}")

if __name__ == '__main__':
    main()
