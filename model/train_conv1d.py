# Importações
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Carregar os dados
df = pd.read_csv('datapose.csv')
df = df[['LEFT_SHOULDER.x', 'LEFT_SHOULDER.y', 'LEFT_SHOULDER.z',
       'LEFT_SHOULDER.visibility', 'RIGHT_SHOULDER.x', 'RIGHT_SHOULDER.y',
       'RIGHT_SHOULDER.z', 'RIGHT_SHOULDER.visibility', 'LEFT_ELBOW.x',
       'LEFT_ELBOW.y', 'LEFT_ELBOW.z', 'LEFT_ELBOW.visibility',
       'RIGHT_ELBOW.x', 'RIGHT_ELBOW.y', 'RIGHT_ELBOW.z',
       'RIGHT_ELBOW.visibility', 'LEFT_WRIST.x', 'LEFT_WRIST.y',
       'LEFT_WRIST.z', 'LEFT_WRIST.visibility', 'RIGHT_WRIST.x',
       'RIGHT_WRIST.y', 'RIGHT_WRIST.z', 'RIGHT_WRIST.visibility',
       'LEFT_PINKY.x', 'LEFT_PINKY.y', 'LEFT_PINKY.z', 'LEFT_PINKY.visibility',
       'RIGHT_PINKY.x', 'RIGHT_PINKY.y', 'RIGHT_PINKY.z',
       'RIGHT_PINKY.visibility', 'LEFT_INDEX.x', 'LEFT_INDEX.y',
       'LEFT_INDEX.z', 'LEFT_INDEX.visibility', 'RIGHT_INDEX.x',
       'RIGHT_INDEX.y', 'RIGHT_INDEX.z', 'RIGHT_INDEX.visibility',
       'LEFT_THUMB.x', 'LEFT_THUMB.y', 'LEFT_THUMB.z', 'LEFT_THUMB.visibility',
       'RIGHT_THUMB.x', 'RIGHT_THUMB.y', 'RIGHT_THUMB.z',
       'RIGHT_THUMB.visibility', 'LEFT_HIP.x', 'LEFT_HIP.y', 'LEFT_HIP.z',
       'LEFT_HIP.visibility', 'RIGHT_HIP.x', 'RIGHT_HIP.y', 'RIGHT_HIP.z',
       'RIGHT_HIP.visibility', 'Label']]

# Separar as features (pontos do mediapipe) e os labels
X = df.iloc[:, :-1].values  # Assumindo que as últimas colunas são os labels
y = df.iloc[:, -1].values - 1  # Subtraindo 1 para converter labels para 0-based

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Criar um Dataset personalizado
class MediapipeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Criar DataLoaders para treinamento e teste
train_dataset = MediapipeDataset(X_train, y_train)
test_dataset = MediapipeDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Rede convolucional
class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (56 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, 12)  #Numero de classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (x.shape[2]))  # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializar o modelo
model = Conv1DNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Função de treinamento
def train_model(model, train_loader, criterion, optimizer, num_epochs=1000):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Adicionar a dimensão do canal
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Treinar o modelo
train_model(model, train_loader, criterion, optimizer)

# Função de avaliação
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')

# Avaliar o modelo
evaluate_model(model, test_loader)

# Caminho para salvar o modelo
model_path = 'conv1d-2.pth'

# Salvar o modelo
torch.save(model.state_dict(), model_path)
print(f'Modelo salvo em {model_path}')
