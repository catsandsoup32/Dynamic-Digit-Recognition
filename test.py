import torch  
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy

from models import CNN, VamsiNN
from dataloader import MathSymbolDataset
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # subtract 0.5 and then divide 0.5 (z-score)
])

test_dataset = MathSymbolDataset(data_dir='data/extracted_images', mode = 'test', transform=transform, seed=42)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN()
model.eval()
model.to(device)
accuracy = Accuracy(task='multiclass', num_classes=81)
model.load_state_dict(torch.load('save_states/CNNmodel12Epoch50.pt', map_location=device, weights_only=True))

running_acc = 0
for images, labels in tqdm(test_loader, desc="testing"):
    images.to(device)
    labels.to(device)
    outputs = model(images)
    running_acc += accuracy(outputs, labels)

print(f"TEST ACC: {running_acc / len(test_loader.dataset) * 32 * 100}")

