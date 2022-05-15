
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch

def cifar10_dataset(path='../data/vision/cifar10/'):
    cifar10_mean = [0.485, 0.456, 0.406]
    cifar10_std = [0.229, 0.224, 0.225]

    cifar10_train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ]
            )
    cifar10_test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ]
            )
    cifar10_train = CIFAR10(path, train=True, transform=cifar10_train_transform)
    cifar10_test = CIFAR10(path, train=False, transform=cifar10_test_transform)
    return cifar10_train, cifar10_test


def save_checkpoint(name, model):
    torch.save(model.state_dict(), f"checkpoints/{name}.th")
    
def save_loss(path, lsts):
    with open(path, "w") as f:
        for lst in lsts:
            for s in lst:
                f.write(str(s) +"\n")
            f.write('\n')
