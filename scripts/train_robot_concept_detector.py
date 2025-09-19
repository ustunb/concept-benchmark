import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from concept_benchmark.ext.fileutils import load
from concept_benchmark.models import FrontEndModel
from concept_benchmark.paths import results_dir
from utils import determine_device


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

device = determine_device()

class RobotConceptClassifier(nn.Module):
    def __init__(self):
        super(RobotConceptClassifier, self).__init__()
        
        # 🧠 1. Shared CNN Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Flattened feature size assuming 128x128 input image
        # 128 -> 64 -> 32 -> 16. So, 64 * 16 * 16
        feature_size = 64 * 28 * 28
        
        # 🎯 2. Specialized Heads
        # Binary concepts (1 output neuron)
        self.head_shape_head = nn.Linear(feature_size, 1)
        self.body_shape_head = nn.Linear(feature_size, 1)
        self.has_knees_head = nn.Linear(feature_size, 1)
        self.has_elbows_head = nn.Linear(feature_size, 1)
        self.has_antennae_head = nn.Linear(feature_size, 1)
        self.ears_shape_head = nn.Linear(feature_size, 1)
        self.mouth_type_head = nn.Linear(feature_size, 1)
        
        # Multi-class concepts (6 output neurons)
        self.hand_shape_head = nn.Linear(feature_size, 1)
        self.foot_shape_head = nn.Linear(feature_size, 1)

    def forward(self, x):
        # Pass input through the shared backbone and flatten
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        # Get predictions from each head
        # We'll apply sigmoid/softmax in the loss function for better numerical stability
        outputs = {
            'head_shape': self.head_shape_head(features),
            'body_shape': self.body_shape_head(features),
            'has_knees': self.has_knees_head(features),
            'has_elbows': self.has_elbows_head(features),
            'has_antennae': self.has_antennae_head(features),
            'ears_shape': self.ears_shape_head(features),
            'mouth_type': self.mouth_type_head(features),
            'hand_shape': self.hand_shape_head(features),
            'foot_shape': self.foot_shape_head(features)
        }
        
        return outputs

model = RobotConceptClassifier()
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

loss_map = {
    'head_shape': bce_loss,
    'body_shape': bce_loss,
    'has_knees': bce_loss,
    'has_elbows': bce_loss,
    'has_antennae': bce_loss,
    'ears_shape': bce_loss,
    'mouth_type': bce_loss,
    'hand_shape': bce_loss,
    'foot_shape': bce_loss
}

loader_config = {
    'batch_size': 32,
    'num_workers': 12,
    'pin_memory': True
}

train_loader = data.training.loader(shuffle=True, **loader_config)

for epoch in tqdm(range(10)): 
    for boards, labels, _ in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        model.to(device)
        boards, labels = boards.to(device), labels.to(device)
        outputs = model(boards)

        loss = 0
        for feat, loss_fn in loss_map.items():
            c_idx = data.meta['concepts'].index(feat)
            # if feat in ['hand_shape', 'foot_shape']:
            #     loss += loss_fn(outputs[feat], labels[:, c_idx].long())
            # else:
            loss_val = loss_fn(outputs[feat].squeeze(), labels[:, c_idx].float())
            mult = 3 if feat == 'has_antennae' else 1
            loss += mult * loss_val

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

# print accuracies for each concept
def evaluate_model(model, data_loader):
    """returns accuracy of the model on the given data_loader"""
    with torch.no_grad():
        model.eval()
        preds = []
        for boards, labels, _ in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            output_dict = model(boards)
            outputs = np.vstack(
                [torch.sigmoid(output_dict[feat].squeeze().cpu()).numpy() for feat in data.meta['concepts']]
            ).T
            predicted = (outputs > 0.5)
            batch_correct = (predicted == labels.cpu().numpy())
            preds.append(batch_correct)

    preds = np.vstack(preds)
    accuracy = preds.mean(axis=0)  # Mean accuracy for each concept
    return accuracy

def predict_model(model, data_loader):
    """returns predictions of the model on the given data_loader"""
    with torch.no_grad():
        model.eval()
        preds = []
        for boards, labels, _ in data_loader:
            boards, labels = boards.to(device), labels.to(device)
            outputs = model(boards)
            predicted = (outputs > 0.5).cpu().numpy()
            preds.append(predicted)
    
    preds = np.vstack(preds)
    return preds


train_loader = data.training.loader(shuffle=False, **loader_config)
valid_loader = data.validation.loader(shuffle=False, **loader_config)
test_loader = data.test.loader(shuffle=False, **loader_config)
train_accuracy = evaluate_model(model, train_loader)
valid_accuracy = evaluate_model(model, valid_loader)
test_accuracy = evaluate_model(model, test_loader)

# Concept accuracies
fe = FrontEndModel()
fe.fit(data.training.C, data.training.y)

train_c_pred = predict_model(model, train_loader)
valid_c_pred = predict_model(model, valid_loader)
test_c_pred = predict_model(model, test_loader)

data_embed = data.embed(model, device=device)

train_y_pred = fe.predict(train_c_pred)
valid_y_pred = fe.predict(valid_c_pred)
test_y_pred = fe.predict(test_c_pred)

train_acc = (train_y_pred == data.training.y).mean()
valid_acc = (valid_y_pred == data.validation.y).mean()
test_acc = (test_y_pred == data.test.y).mean()