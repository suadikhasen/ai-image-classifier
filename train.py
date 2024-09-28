import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def get_input_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    
    # Add a required argument (the user must provide this)
    parser.add_argument('data_dir', type=str, help='Path to the folder containing your dataset.')
    
    # Add optional arguments (the user can choose to provide these or not)
    parser.add_argument('--hidden_units', type=int, default=4096, 
                        help='Number of neurons in the hidden layer. Default is 4096.')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='The type of pre-trained model to use. Options are vgg16 or vgg13. Default is vgg16.')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='How fast the model learns. Default is 0.001.')
    parser.add_argument('--save_dir', type=str, default='.', 
                        help='Folder to save your trained model. Default is the current folder.')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of times to train on the entire dataset. Default is 5.')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use this flag if you want to use a GPU for faster training.')
    
    # Process the arguments and return them
    return parser.parse_args()
def load_data(data_dir):
    # Set up the paths for training and validation data
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    # Define how to prepare the training images
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406],  # Normalize the image
                             [0.229, 0.224, 0.225])
    ])
    
    # Define how to prepare the validation images
    valid_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the image
        transforms.CenterCrop(224),  # Crop the center of the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406],  # Normalize the image
                             [0.229, 0.224, 0.225])
    ])
    
    # Load the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Create data loaders for easy batching
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    # Return the datasets and data loaders
    return train_dataset, train_loader, valid_loader

def build_model(arch, hidden_units):
    # Load a pre-trained network
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print(f"Sorry, we don't support the {arch} architecture.")
        exit()
    
    # Freeze the parameters so we don't change the pre-trained part
    for param in model.parameters():
        param.requires_grad = False
    
    # Create a new classifier for our specific problem
    input_features = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the old classifier with our new one
    model.classifier = classifier
    
    return model
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    # Initialize variables to keep track of training progress
    total_steps = 0
    total_train_loss = 0
    print_frequency = 40
    
    # Loop through each epoch
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        
        model.train()
        
        for inputs, labels in train_loader:
            total_steps += 1
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            optimizer.step()
            
            # Keep track of the training loss
            total_train_loss += loss.item()
            
            # Print progress and evaluate the model periodically
            if total_steps % print_frequency == 0:
                print(f"  Batch {total_steps}: Evaluating model...")
                
                model.eval()
                
                total_valid_loss = 0
                total_accuracy = 0
                
                # Disable gradient computation for validation
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        
                        batch_loss = criterion(outputs, labels)
                        total_valid_loss += batch_loss.item()
                        
                        # Compute the accuracy
                        probabilities = torch.exp(outputs)
                        top_probability, top_class = probabilities.topk(1, dim=1)
                        correct_predictions = top_class == labels.view(*top_class.shape)
                        accuracy = torch.mean(correct_predictions.type(torch.FloatTensor)).item()
                        total_accuracy += accuracy
                
                # Print the results
                avg_train_loss = total_train_loss / print_frequency
                avg_valid_loss = total_valid_loss / len(valid_loader)
                avg_accuracy = total_accuracy / len(valid_loader)
                
                print(f"    Epoch {epoch+1}/{epochs}")
                print(f"    Average Training Loss: {avg_train_loss:.3f}")
                print(f"    Average Validation Loss: {avg_valid_loss:.3f}")
                print(f"    Average Validation Accuracy: {avg_accuracy:.3f}")
                
                total_train_loss = 0
                
                model.train()
        
    print("Training completed!")

def save_checkpoint(model, train_dataset, save_dir, arch, hidden_units, learning_rate, epochs,optimizer):
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {
        'architecture': arch,  
        'classifier': model.classifier,  
        'state_dict': model.state_dict(),  
        'class_to_idx': model.class_to_idx, 
        'optimizer_state_dict': optimizer.state_dict(),  
        'epochs': epochs,  
        'learning_rate': learning_rate,  
        'hidden_units': hidden_units  
        }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save our checkpoint to a file
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    args = get_input_arguments()
    
    # Choose whether to use GPU or CPU
    if args.gpu and torch.cuda.is_available():
        print("Using GPU for training")
        device = torch.device('cuda')
    else:
        print("Using CPU for training")
        device = torch.device('cpu')
    
    print("Loading and preparing the data...")
    train_dataset, train_loader, valid_loader = load_data(args.data_dir)
    
    model = build_model(args.arch, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    model.to(device)
    
    print("Starting to train the model...")
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    
    print("Saving the trained model...")
    save_checkpoint(model, train_dataset, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs,optimizer)
    
    print("All done! The model has been trained and saved.")

if __name__ == '__main__':
    main()



