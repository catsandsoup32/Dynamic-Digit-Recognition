import torch  
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms as transforms
from torchmetrics import Accuracy

from NEW_models import CNN_16
from NEW_dataloader import MathSymbolDataset
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # subtract 0.5 and then divide 0.5 (z-score)
])
def main():
    test_dataset = MathSymbolDataset(data_dir='data/extracted_images_new', mode = 'test', transform=transform, seed=21)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CNN_16()
    model.eval()
    model.to(device)
    accuracy = Accuracy(task='multiclass', num_classes=72).to(device)
    model.load_state_dict(torch.load('NEW_save_states/CNNmodel18Epoch105.pt', map_location=device, weights_only=True))

    running_acc = 0
    for images, labels in tqdm(test_loader, desc="testing"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        running_acc += accuracy(outputs, labels)

    print(f"TEST ACC: {running_acc / len(test_loader.dataset) * 32 * 100}")


if __name__ == '__main__':
    main()