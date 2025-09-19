from concept_benchmark.ext.fileutils import load
from concept_benchmark.paths import results_dir
from torchvision import transforms

def get_dataset_path(**settings) -> str:
    return results_dir / f"robot_{settings['data_type']}.data"

settings = {
    'samples_per_instance': 1,
    'draw': False,
    'output_directory': results_dir / 'robots_large',
    'concepts': {
        'head_shape': ['square', 'round'],
        'body_shape': ['square', 'round'],
        'has_knees': ['false', 'true'],
        'has_elbows': ['false', 'true'],
        'has_antennae': ['false', 'true'],
        'ears_shape': ['square', 'triangle'],
        'mouth_type': ['closed', 'open'],
        'hand_shape': ['round_circle', 'round_oval', 'round_oval2',
                        'edgy_triangle', 'edgy_square', 'edgy_trapezoid'],
        'foot_shape': ['flat_4sided', 'flat_5sided', 'flat_lshaped',
                        'pointy_3sided', 'pointy_4sided', 'pointy_6sided'],
    },
    'spurious_features': ['has_elbows', 'hand_shape'],  # features that do not appear in the catalog + color
    'model': "'glorp' if (int(row['body_shape']=='square') + int(row['foot_shape']=='pointy') - 2 >= 0) else 'drent'",
    'model_type': 'deterministic', 
    'size': 'large',  
    'color_mode': 'color',  
    'data_type': 'image'
}

data = load(get_dataset_path(**settings))
IMG_SIZE = 224  # use 384 if you switch to a 384 ViT

tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

data.transform = tf
data.split(fold_id='K05N01', fold_num_validation=4, fold_num_test=5)


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import determine_device

device = determine_device()

class RobotClassifierCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(RobotClassifierCNN, self).__init__()
        
        # --- Feature Extractor ---
        # Input images are assumed to be 3-channel RGB
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves the dimensions again
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Classifier Head ---
        # The input size to the linear layer depends on the input image size.
        # Let's assume input images are 128x128 pixels.
        # After 3 pooling layers, the size becomes 128 -> 64 -> 32 -> 16.
        # So the flattened size is 64 channels * 16 * 16 pixels.
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through the feature extractor
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the feature maps for the classifier
        x = torch.flatten(x, 1)
        
        # Pass through the classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # For binary classification, we apply a sigmoid function to the output
        return torch.sigmoid(x)

# --- Example Usage ---
# Create an instance of the model
model = RobotClassifierCNN()
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loader_config = {
    'batch_size': 32,
    'num_workers': 12,
    'pin_memory': True
}

train_loader = data.training.loader(shuffle=True, **loader_config)

for epoch in tqdm(range(10)): 
    for boards, _, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        model.to(device)
        boards, labels = boards.to(device), labels.to(device)
        outputs = model(boards)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()


# print accuracies
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for boards, _, labels in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predicted = (outputs.squeeze() >= 0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

valid_loader = data.validation.loader(shuffle=False, **loader_config)
test_loader = data.test.loader(shuffle=False, **loader_config)
# train_accuracy = evaluate_model(model, train_loader)
valid_accuracy = evaluate_model(model, valid_loader)
test_accuracy = evaluate_model(model, test_loader)
print(f"Validation Accuracy: {valid_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")