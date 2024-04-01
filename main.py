from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from Classifier import CNN
from Preprocesser import ImagePreprocessor


def train(params={},
          preprocessed_folder="./data/preprocessed_images",
          writer=None,
          device=torch.device('cpu'),
          batch_size=15,
          num_epochs=15,
          weight=[1]):
    # Initialize the model
    model = CNN(params=params).to(device)

    # Define loss function and optimizer
    weight_tensor = torch.tensor(weight).to(device) # TODO: Implemetn
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor).to(device)  # Binary Cross Entropy with Logits Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=1)

    # Define transforms for data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Load the dataset
    train_dataset = ImageFolder(root=preprocessed_folder + '/train', transform=transform)
    validation_dataset = ImageFolder(root=preprocessed_folder + '/train', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    losses = []
    best_loss = np.inf
    best_model = None
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()  # Squeeze the output
            labels = labels.to(device).squeeze().float()  # Squeeze the labels and convert to float

            # print("Outputs: ", outputs)
            # print("Labels: ", labels)

            loss = criterion(outputs, labels)  # Squeeze the output and convert labels to float

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss)
        writer.add_scalar('Loss/train', running_loss, epoch)

        # Validation
        running_loss= 0.0
        for i, (inputs, labels) in enumerate(validation_loader, 0):
            outputs = model(inputs).squeeze()
            labels = labels.to(device).squeeze().float()
            running_loss += criterion(outputs, labels).item()
        writer.add_scalar('Loss/validation', np.sum(running_loss), epoch)
        if np.sum(running_loss) < best_loss:
            best_loss = np.sum(running_loss)
            best_model = model

    return best_model


def test(model, writer=None, device=torch.device('cpu'), batch_size=15):
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    test_dataset = ImageFolder(root="./data/preprocessed_images/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    predictions = []
    truth = []
    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = model(images).to(device)
            predicted = torch.round(torch.sigmoid(outputs))
            for pred in predicted.tolist():
                predictions.append(pred)
            for label in labels.tolist():
                truth.append(label)

    print("Predictions: ", predictions)
    print("Truth: ", truth)

    conf_mat = confusion_matrix(truth, predictions)

    print("Conf_mat: ", conf_mat)

    img_conf_mat = pd.DataFrame(conf_mat / np.sum(conf_mat, axis=1)[:, None],
                                index=['Actual Negative (0)', 'Actual Positive (1)'],
                                columns=['Predicted Negative (0)', 'Predicted Positive (1)'])

    # Plot the heatmap
    plt.figure()
    sn.heatmap(img_conf_mat, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    # Convert the Matplotlib figure to a PIL image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = Image.open(buffer)

    # Convert the image to a PyTorch tensor
    img = transforms.ToTensor()(img)

    # Add the image to the writer (assuming 'writer' is a SummaryWriter object)
    writer.add_image('Confusion Matrix Heatmap TESTING', img)


def main(PREPROCESSING, TRAINING):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    input_folder = "./data/facial palsy patients"
    preprocessed_folder = "./data/preprocessed_images"
    writer = SummaryWriter()
    batch_size = 15

    #  PREPROCESSING

    if PREPROCESSING:
        preprocessor = ImagePreprocessor(input_folder, preprocessed_folder)
        preprocessor.preprocess_images()

    # TRAINING

    if TRAINING:
        params_model = {
            "shape_in": (3, 256, 256),
            "initial_filters": 8,
            "num_fc1": 100,
            "dropout_rate": 0.15,
            "num_classes": 1}

        model = train(writer=writer,
                      device=device,
                      batch_size=batch_size,
                      params=params_model,
                      weight=[30/146])
        test(model, writer=writer, device=device, batch_size=batch_size)


if __name__ == "__main__":
    PREPROCESSING = False
    TRAINING = True

    main(PREPROCESSING, TRAINING)
