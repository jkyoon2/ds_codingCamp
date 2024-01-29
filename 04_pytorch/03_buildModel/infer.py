import torch 
from archs import SimpleVGG
from PIL import Image
from torchvision import transforms

# Set device for computation (CUDA if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for inference")

# Initialize model
model = SimpleVGG(3)

# Load model weights from the checkpoint
model.load_state_dict(torch.load('outputs/checkpoint.pth'))

# Set the model to evaluation mode
model.eval()

# Move the model to the specified device
model.to(device)

# Define the transformation to be applied on the input image
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

# Load the image
image = Image.open('inference_image.jpg')

# Apply the transformations on the image
image = transform(image)

# Add a batch dimension since PyTorch treats all inputs as batches
image = image.unsqueeze(0)

# Move the image to the specified device
image = image.to(device)

# Perform inference on the image using the model
outputs = model(image)

# Apply softmax function to convert model's output to probabilities
probs = torch.nn.functional.softmax(outputs, dim=1)

# Get the class with the highest probability
_, preds = torch.max(probs, dim=1)

# Print the predicted class
print(f"The model predicts class {preds.item()}")
