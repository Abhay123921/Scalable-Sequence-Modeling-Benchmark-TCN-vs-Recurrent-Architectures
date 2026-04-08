import torch
from torchvision import datasets, transforms

def get_seq_mnist(batch_size=64):

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)        # [B,1,28,28]
        images = images.view(len(images), -1, 1)  # [B,784,1]
        images = (images - 0.5) / 0.5   # normalize to [-1,1]
        labels = torch.tensor(labels)
        return images, labels

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader