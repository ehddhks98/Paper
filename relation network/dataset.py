from PIL import Image
from torchvision import transforms
from torchvision.datasets import Omniglot
from sampler import BatchSampler, custom_collate_fn
from torch.utils.data import DataLoader
from functools import partial


def get_dataset(image_size):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_set = Omniglot(
        root="omniglot", transform=transform, background=True, download=True
    )
    test_set = Omniglot(
        root="omniglot", transform=transform, background=False, download=True
    )

    return train_set, test_set


def get_dataloader(
    dataset,
    n_classes,
    n_support,
    n_query,
    n_batches,
    batch_sampler=BatchSampler,
    collate_fn=custom_collate_fn,
):
    batch_sampler = batch_sampler(
        dataset,
        n_classes=n_classes,
        n_support=n_support,
        n_query=n_query,
        n_batches=n_batches,
    )
    collate_fn = partial(collate_fn, n_support=n_support, n_query=n_query)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    return dataloader
