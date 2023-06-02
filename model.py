import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Net(nn.Module):
    
    """
    Our Neural Network class
    
    ...
    Attributes
    ----------
    conv1, conv2, conv3, conv4: Convolution Layers
    fc1, fc2: Fully Connected Layers
    
    Methods
    -------
    forward(x)
        Calculates network prediction on input x
    """

    def __init__(self):
        
        """
        Initializes the structue
        """

        super(Net, self).__init__()
        # r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out: 26, j_out:1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out: 24, j_out:1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # r_in:5, n_in:24, j_in:1, s:2, r_out:7, n_out: 11.5 >> 12,  j_out: 2
        ## MaxPool2D
        # r_in:7, n_in: 12, j_in: 2, r_out:11, n_out: 10, j_out: 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # r_in:11, n_in: 10, j_in: 2, s:1, r_out:15, n_out: 8, j_out:2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # r_in:15, n_in: 8, j_in:2, s:2, r_out: 19, n_out: 3.5 >> 4, j_out: 4
        ## MaxPool2D
        # n_in: 256*4*4, n_out: 50
        self.fc1 = nn.Linear(4096, 50)
        # n_in: 50, n_out: 10
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        """
        Defines the forward pass for the network
        :param x (Torch tensor): Input batch 
        """
        x = F.relu(self.conv1(x)) # CONV > RELU
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # CONV > POOL > RELU
        x = F.relu(self.conv3(x)) # CONV > RELU
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # CONV > POOL > RELU
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x)) # FC > RELU
        x = self.fc2(x) # FC
        return F.log_softmax(x, dim=1) # Returns log softmax along columns for each row

def GetCorrectPredCount(pPrediction, pLabels):
  
    """
    Function that calculate the number of correct predictions 
    :pPrediction param: Neural Network Predictions
    :pLabels param: Ground Truth Labels
    """

    # We first get tensor of indices corresponding to highest logmax prediction for each image
    # We then compare this with a tensor of ground truth values. True if equal, False if not.
    # We sum the tensor up to get count of correct predictions and return value as an integer
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, train_acc, train_losses):
  
  """
  Function that trains our neural network model
  :param model: Neural Network model
  :param device: Device where training occurs ('cuda' or 'cpu')
  :param train_loader: Dataloader for training data
  :param optimizer: Optimizer for applying gradient updates
  :param train_acc: List that stores training accuracy values
  :param train_losses: List that stores training loss values
  """
  
  # Set the model to train mode
  model.train()

  # Initialize the progress bar to display information in each step
  pbar = tqdm(train_loader)

  # Initialize batch loss tracker, # of correct preds & # of images processed
  train_loss = 0
  correct = 0
  processed = 0

  # Iterate through each batch of data along with corresponding labels
  for batch_idx, (data, target) in enumerate(pbar):
    
    # Move the data batch and the labels to CUDA if available
    data, target = data.to(device), target.to(device)
    
    # Flush out the gradient update values from the previous iteration 
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate average batch loss and increment
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Increment # of correct predictions for the batch
    correct += GetCorrectPredCount(pred, target)
    
    # Increment # of processed images
    processed += len(data)

    # Print batch ID and train loss & accuracy so far
    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  # Append epoch accuracy to a list for visualisation
  train_acc.append(100. * correct/processed)

  # Append average loss in the epoch
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, test_acc, test_losses):

    """
    Function for testing a trained model on test data
    :param model: Neural Network model
    :param device: Device where training occurs ('cuda' or 'cpu')
    :param test_loader: Dataloader for test data
    :param test_acc: List that stores test accuracy values
    :param test_losses: List that stores test loss values
    """
    
    # Set the model to eval mode
    model.eval()

    # Initialize loss tracker variable and total # of correct predictions
    test_loss = 0
    correct = 0

    # Disable backpropagation
    with torch.no_grad():
        
        # Iterate through each batch and target labels
        for batch_idx, (data, target) in enumerate(test_loader):
            
            # Move the data batch and target labels to the selected device
            data, target = data.to(device), target.to(device)

            # Get the model output for the batch of data
            output = model(data)
            
            # Increment total loss for this batch of data
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # Get the number of correct predictions
            correct += GetCorrectPredCount(output, target)

    # Get the average test loss
    test_loss /= len(test_loader.dataset)

    # Append average test batch accuracy for the epoch
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    # Append average test loss for the epoch
    test_losses.append(test_loss)

    # Print the average loss and accuracy on the test set
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
     
def train_test_run(model, device, train_loader, test_loader, num_epochs,
                   train_acc, train_losses, test_acc, test_losses):

    """
    Function to train and test our model for num_epochs
    :param model: Neural Network Model
    :param device: Device where training occurs ('cuda' or 'cpu')
    :param train_loader: Dataloader for training data
    :param test_loader: Dataloader for test data
    :param num_epochs: Number of epochs
    """


    # Define a SGD optimizer with selected parameter values
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define a learning rate scheduler that decays lr of underlying optimizer
    # by 10% every 15 epochs 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

    # Iterating over num_epochs
    for epoch in range(1, num_epochs+1):

        print(f'Epoch {epoch}')
        
        # Invoke training loop
        train(model, device, train_loader, optimizer, train_acc, train_losses)

        # Invoke test loop
        test(model, device, test_loader, test_acc, test_losses)
        
        # Invoke the scheduler
        scheduler.step()