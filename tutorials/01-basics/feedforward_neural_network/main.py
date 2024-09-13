import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #inisiasi input layer
        
        #inisiasi output layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        
        self.fc10 = nn.Linear(hidden_size, num_classes) #inisiasi output layer
        self.relu = nn.ReLU() #fungsi aktivasi ReLU
        self.sigmoid = nn.Sigmoid() #fungsi aktifasi sigmoid
    
    def forward(self, x):
        out = self.fc1(x) #memasukkan data ke dalam input layer
        out = self.relu(out) #melakukan fungsi aktivasi untuk hasil data di input layer
        
         #memasukkan data ke dalam hidden layer dan melakukan fungsi aktivasi
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out = self.sigmoid(out)
        
        out = self.fc5(out)
        out = self.relu(out)
        
        out = self.fc6(out)
        out = self.sigmoid(out)
        
        out = self.fc7(out)
        out = self.relu(out)
        
        out = self.fc8(out)
        out = self.sigmoid(out)
        
        out = self.fc9(out)
        out = self.relu(out)
        
        #memasukkan data ke dalam output layer
        out = self.fc10(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device) #pembuatan model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
loss_list = []
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass (hitung prediksi model)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize (mengubah gradien ke 0 sebelum menjalankan backward dan mengubah weight berdasarkan gradien)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

plt.figure(figsize=(10, 5))
plt.plot(loss_list, label='default')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss table')
plt.legend()
plt.grid(True)
plt.show()