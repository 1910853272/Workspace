import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 50

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

model = CNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

Epoch = 5

# Initialize lists to store accuracy and loss for plotting
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Initialize lists for batch-wise loss and accuracy (recording every 100 batches)
batch_losses = []
batch_accuracies = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # Track accuracy
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

        # Record every 100th batch loss and accuracy
        if batch % 100 == 0:
            batch_loss = loss.item()
            batch_accuracy = (pred.argmax(1) == y).sum().item() / len(y)
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)

            # Print loss and accuracy for each 100th batch
            print(f"Batch {batch+1}: Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy*100:.2f}%")

    # Calculate average loss and accuracy for the epoch
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total

    # Print average loss and accuracy at the end of the epoch
    print(f"Epoch End: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {accuracy*100:.2f}%")

    return avg_loss, accuracy

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, accuracy

for t in range(Epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    # Train the model
    train_loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
    # Test the model
    test_loss, test_accuracy = test(test_dataloader, model, loss_fn)

    # Store the results for visualization
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the loss and accuracy for the entire training
epochs = range(1, Epoch + 1)

# Plot loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# Save the figure as an image file
plt.tight_layout()
plt.savefig('training_results.png')

# Display the plot
plt.show()

# Plot batch-wise loss and accuracy (every 100th batch)
plt.figure(figsize=(12, 6))

# Plot loss per batch
plt.subplot(1, 2, 1)
plt.plot(batch_losses, label='Batch Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Loss per 100 Batches')
plt.legend()

# Plot accuracy per batch
plt.subplot(1, 2, 2)
plt.plot(batch_accuracies, label='Batch Accuracy')
plt.xlabel('Batch Number')
plt.ylabel('Accuracy')
plt.title('Accuracy per 100 Batches')
plt.legend()

# Save the figure as an image file
plt.tight_layout()
plt.savefig('batch_results.png')

# Display the plot
plt.show()

print("Done!")
