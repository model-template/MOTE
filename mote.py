import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class FaceTemplateClassifier1(nn.Module):
    def __init__(self, input_size=512, h1=128, h2=64, h3=16, output_size=1, dropout_rate=0.5):
        super(FaceTemplateClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(h2, h3)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(h3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class FaceTemplateClassifier2(nn.Module):
    def __init__(self, input_size=512, h1=256, h2=128, output_size=1, dropout_rate=0.5):
        super(FaceTemplateClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(h2, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class FaceTemplateClassifierSmall(nn.Module):
    def __init__(self, input_size=512, h1=64, h2=16, output_size=1, dropout_rate=0.5):
        super(FaceTemplateClassifierSmall, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(h2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class RandomFaceTemplateClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[128, 64, 16], output_size=1, dropout_rate=0.5, random_range=10):
        super(RandomFaceTemplateClassifier, self).__init__()
        random_hidden_sizes = [
            size + random.randint(-random_range, random_range)
            for size in hidden_sizes
        ]
        layer_sizes = [input_size] + random_hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU() if i < len(layer_sizes) - 2 else nn.Identity(),
                nn.Dropout(dropout_rate) if i < len(layer_sizes) - 2 else nn.Identity()
            ))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
