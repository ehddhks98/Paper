from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader


def get_dataloaders(train_set, test_set, n_way, n_shot, n_query, n_training_episodes, n_evaluation_tasks):
    train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_training_episodes)
    test_sampler = TaskSampler(test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_evaluation_tasks)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=12, pin_memory=True, collate_fn=train_sampler.episodic_collate_fn)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=12, pin_memory=True, collate_fn=test_sampler.episodic_collate_fn)

    return train_loader, test_loader