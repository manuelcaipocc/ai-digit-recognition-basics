from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle
import torch
from PIL import Image
from torchvision import transforms
import cv2
# Configure Loguru to save logs to a file
logger.add("mnist_logs.log", format="{time} {level} {message}", level="INFO", rotation="10 KB", compression="zip")


def show_image_with_label(data, index):
    """
    Function to visualize an image and its label from a tensor dataset.
    
    Args:
    - data (list): List of tuples where each tuple contains an image tensor and its label.
    - index (int): Index of the element to visualize.
    
    Returns:
    - None: Displays the image with its label in the title.
    """
    # Get the image and label from the provided index
    image, label = data[index]
    
    # Convert the image tensor to a numpy array for visualization with matplotlib
    image = image.numpy()

    # Create the plot with aspect ratio 'auto' to preserve the original dimensions
    plt.title(f'Image from Dataset - Label: {label}')
    plt.imshow(image, cmap='gray')


def load_mnist_images(image_file):
    """
    Loads images from a MNIST binary file and returns the data.

    :param image_file: Path to the MNIST image file.
    :return: A tuple (num_images, num_rows, num_cols, images), where:
        - num_images: Total number of images.
        - num_rows: Number of rows per image.
        - num_cols: Number of columns per image.
        - images: Numpy array containing the images.
    """
    try:
        with open(image_file, 'rb') as f:
            magic_number = int.from_bytes(f.read(4), 'big')  # Magic number
            num_images = int.from_bytes(f.read(4), 'big')    # Number of images
            num_rows = int.from_bytes(f.read(4), 'big')      # Rows per image
            num_cols = int.from_bytes(f.read(4), 'big')      # Columns per image
            
            # Read the image data
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape((num_images, num_rows, num_cols))
        
        logger.info(f"File '{image_file}' loaded successfully.")
        logger.info(f"Number of images: {num_images}, Dimensions: {num_rows}x{num_cols}")
        return num_images, num_rows, num_cols, images
    except FileNotFoundError:
        logger.error(f"File '{image_file}' not found.")
        raise
    except Exception as e:
        logger.exception(f"An error occurred while loading the file '{image_file}': {e}")
        raise

def display_image(images, index):
    """
    Displays a specific image from a set of images.

    :param images: Numpy array containing the images.
    :param index: Index of the image to display.
    """
    try:
        if 0 <= index < images.shape[0]:
            plt.imshow(images[index], cmap='gray')
            plt.title(f"Image #{index}")
            #plt.axis('off')
            plt.show()
            logger.info(f"Displayed image at index {index}.")
        else:
            logger.warning(f"Index {index} is out of range. Valid range is 0 to {images.shape[0] - 1}.")
            print(f"Index out of range. Please choose a value between 0 and {images.shape[0] - 1}.")
    except Exception as e:
        logger.exception(f"An error occurred while displaying the image at index {index}: {e}")
        raise

def get_total_batches(dataloader):
    return len(dataloader)

# Function 2: Total number of images
def get_total_images(dataloader):
    return len(dataloader.dataset)

# Function 3: Inspect a specific batch and index
def inspect_batch(dataloader, batch_idx, img_idx):
    # Ensure batch_idx is within range
    if batch_idx >= len(dataloader):
        raise ValueError(f"Invalid batch index {batch_idx}. Total batches: {len(dataloader)}")
    
    # Fetch the specific batch
    data_iter = iter(dataloader)
    for i in range(batch_idx + 1):
        inputs, labels = next(data_iter)
    
    # Ensure img_idx is within range
    if img_idx >= len(inputs):
        raise ValueError(f"Invalid image index {img_idx}. Batch size: {len(inputs)}")
    
    # Display information
    print(f"Batch {batch_idx}, Image {img_idx}")
    print(f"Input size: {inputs[img_idx].size()}")  # Dimensions of the image tensor
    print(f"Label: {labels[img_idx].item()}")      # Corresponding label
    print(f"Label format: {type(labels)}")        # Type of the labels (tensor)

    # Visualize the image and its label
    plt.imshow(inputs[img_idx].squeeze(), cmap='gray')
    plt.title(f"Label: {labels[img_idx].item()}")
    plt.axis('off')
    plt.show()

def visualize_transformed_image(loader, batch_number, image_index):
    """
    Visualize a transformed image and its label.

    Parameters:
    - loader: The DataLoader object.
    - batch_number: The batch number to retrieve.
    - image_index: The index of the image within the selected batch.
    """
    # Convert the loader to a list for indexed access
    batches = list(loader)

    # Ensure the batch_number is within range
    if batch_number < 0 or batch_number >= len(batches):
        print(f"Error: Batch number {batch_number} is out of range. Total batches: {len(batches)}")
        return

    # Retrieve the specified batch
    images, labels = batches[batch_number]

    # Ensure the image_index is within range
    if image_index < 0 or image_index >= images.size(0):
        print(f"Error: Image index {image_index} is out of range. Total images in batch: {images.size(0)}")
        return

    # Get the specific image and label
    image = images[image_index]
    label = labels[image_index]

    # Denormalize the image for visualization
    image = image * 0.5 + 0.5  # Reverse normalization (assuming mean=0.5, std=0.5)
    image = image.permute(1, 2, 0).numpy()  # Convert to H x W x C format for matplotlib

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.show()

import torch

def load_and_evaluate(model, filepath, testloader, device):
    """
    Load model weights from a .pth file and evaluate accuracy.

    Parameters:
    - model: PyTorch model.
    - filepath: Path to the .pth file containing the saved weights.
    - testloader: DataLoader with the test data.
    - device: Device ('cpu' or 'cuda').

    Returns:
    - accuracy: Model accuracy on the test dataset.
    """
    # Load model weights
    try:
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.to(device)
        print(f"Model successfully loaded from {filepath}.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy

import torch
import matplotlib.pyplot as plt

def display_predictions_grid(model, dataloader, device):
    """
    Display a 4x4 grid of predictions, true labels, and images.

    Parameters:
    - model: PyTorch model.
    - dataloader: DataLoader (testloader or trainloader).
    - device: Device ('cpu' or 'cuda').

    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Initialize plot
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Predictions vs True Labels", fontsize=16)

    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Display 16 images, labels, and predictions
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            break

        # Move image to CPU for display
        image = images[idx].cpu().squeeze()  # Remove channel dimension if present

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Pred: {predictions[idx].item()}\nTrue: {labels[idx].item()}")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.show()

def load_and_transform_images(image_folder="images"):
    """
    Loads images from the folder and transforms them into PyTorch tensors,
    assigning a label based on the filename.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        #transforms.Resize((8, 8)),  # Resize to 8x8
        transforms.ToTensor()  # Convert to tensor
    ])
    
    data = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            label = int(filename.split(".")[0])  # Extract label from filename
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path)
            tensor_image = transform(image)
            tensor_image = tensor_image.squeeze(0)  # Remove extra channel dimension
            for _ in range(1000):  # Duplicate each tuple 100 times
                data.append((tensor_image, label))
    
    return data

def load_pkl_file(pkl_file="C64_Ziffern_Daten.pkl"):
    """
    Loads the existing pickle file and returns the data.
    """
    with open(pkl_file, "rb") as file:
        data = pickle.load(file)
    return data

def update_pkl_file(new_data, pkl_file="C64_Ziffern_Daten.pkl", output_file=None):
    """
    Adds new transformed images to the pickle file and saves the result.
    Allows saving the file under a new name if output_file is provided.
    """
    existing_data = load_pkl_file(pkl_file)
    combined_data = existing_data + new_data
    
    save_path = output_file if output_file else pkl_file
    with open(save_path, "wb") as file:
        pickle.dump(combined_data, file)
    
    print(f"Total tuples after update: {len(combined_data)}")
    print(f"File saved as: {save_path}")
    return len(combined_data)

def live_number_recognition(model, device):
    """
    Real-time number recognition with preprocessing to handle illumination variations.

    Parameters:
    - model: PyTorch model for digit recognition.
    - device: Device ('cpu' or 'cuda').
    """
    # Configure the model
    model.to(device)
    model.eval()

    # Capture video from the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Preprocess the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Adaptive Thresholding to handle lighting variations
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Optionally, detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate the thresholded image to separate digits
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Detect contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_images = []
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 50 and w < frame.shape[1] * 0.8:  # Filter small and very large contours
                digit = thresh[y:y + h, x:x + w]
                digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                digit_images.append(digit_resized)
                bounding_boxes.append((x, y, w, h))

        # Sort by x position (left to right)
        sorted_boxes = sorted(zip(bounding_boxes, digit_images), key=lambda b: b[0][0])

        predictions = []
        for box, digit_image in sorted_boxes:
            digit_tensor = torch.tensor(digit_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            digit_tensor = digit_tensor.to(device)

            with torch.no_grad():
                output = model(digit_tensor)
                _, predicted = torch.max(output, 1)
                predictions.append((predicted.item(), box))

        # Combine predictions into a multi-digit number
        combined_number = "".join(str(pred[0]) for pred in predictions)

        # Draw the recognized numbers and the full prediction on the frame
        for prediction, (x, y, w, h) in predictions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the full number at the top of the frame
        cv2.putText(frame, f"Detected: {combined_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the processed frame
        cv2.imshow("Real-Time Number Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

