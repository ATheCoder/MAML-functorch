import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from pathlib import Path
import random
from torchvision.transforms import Compose, Resize, RandomCrop, ColorJitter, RandomHorizontalFlip, InterpolationMode
from torchvision.io import read_image, ImageReadMode
from maml_functorch.utils import collate_fn
from dataclasses import dataclass
from torch.utils.data import DataLoader
import random

class MiniImageNet(Dataset):
    def __init__(self, root_dir, mini_batch_size, ways, shots, query_size, device, data_fold = 'train') -> None:
        super().__init__()
        
        self.data_fold = data_fold
        self.ways = ways
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.query_size = query_size
        self.transform = Compose([
            lambda x: read_image(x, ImageReadMode.RGB),
            Resize((84, 84)),
            lambda x: x / 255.0,
        ])
        self.shots = shots
        
        self.root_path = Path(root_dir)
        
        self.label_to_filename = self.generate_label_to_filename_dict(self.root_path)
        
        self.classes = list(self.label_to_filename.keys())
        
        self.label_to_counter = self.init_label_to_counter(self.classes)
        
        self.tasks = []
        self.cached_task = {}
        
    def init_label_to_counter(self, classes):
        label_to_counter = {}
        
        for way in classes:
            label_to_counter[way] = 0
            
        return label_to_counter

    def generate_label_to_filename_dict(self, root_dir: Path):
        df = pd.read_csv(root_dir / f'{self.data_fold}.csv')
        df = df.reset_index()

        label_to_filename = {}

        for t in df.itertuples():
            file_name, label = t[2], t[3]
            
            if label not in label_to_filename:
                label_to_filename[label] = [file_name]
            else:
                label_to_filename[label].append(file_name)

        return label_to_filename # TODO: You need to shuffle the array as well!
    
    def shuffle_set(self, s, s_l):
        permutations = torch.randperm(s.size(0))
        
        q = s[permutations]
        t = s_l[permutations]
        
        return q, t
    
    def shuffle_task(self, task):
        support_set, support_labels = self.shuffle_set(task['support_set'], task['support_labels'])
        
        query_set, query_labels = self.shuffle_set(task['query_set'], task['query_labels'])
        
        return {'support_set': support_set, 'support_labels': support_labels, 'query_set': query_set, 'query_labels': query_labels}
    
    def preprocess_image_path(self, image_name):
        image_str_path = self.root_path / 'images' / image_name
        
        return str(image_str_path.resolve())
    
    def open_task_images(self, task):
        task['support_set'] = [self.preprocess_image_path(image_name) for image_name in task['support_set']]
        task['query_set'] = [self.preprocess_image_path(image_name) for image_name in task['query_set']]
        
        return task
            
    
    def __getitem__(self, i):
        classes_for_task = random.sample(self.classes, self.ways)
        
        
        
        task = {'support_set': [], 'query_set': [], 'support_labels': []}
        
        for label in classes_for_task:
            for_this_label = random.sample(self.label_to_filename[label], self.shots + self.query_size)
            
            task['query_set'] = task['query_set'] + for_this_label[:self.query_size]
            task['support_set'] = task['support_set'] + for_this_label[self.query_size:]
            
        task['support_labels'] = [num for num in range(self.ways) for _ in range(self.shots)]
        task['query_labels'] = [num for num in range(self.ways) for _ in range(self.query_size)]
        
        task = self.open_task_images(task)
        
        transformed_support_set = torch.stack(self.run_transform(task['support_set']))
        
        transformed_query_set = torch.stack(self.run_transform(task['query_set']))
        
        transformed_support_labels = torch.tensor(task['support_labels'], device=self.device)
        transformed_query_labels = torch.tensor(task['query_labels'], device=self.device)
        
        task = {'support_set': transformed_support_set, 'query_set': transformed_query_set, 'support_labels': transformed_support_labels ,'query_labels': transformed_query_labels}
        
        
        return self.shuffle_task(task)
        
    def __len__(self):
        return 20_000 * 4 * 7
    
    def run_transform(self, samples):
        return [self.transform(s).float() for s in samples]
    
@dataclass(frozen=True)
class DataConfig:
    functional: bool
    ways: int
    shots: int
    batch_size: int
    query_size: int
    
def load_data(args: DataConfig):
    loader_collate_fn = collate_fn if not args.functional else None
    
    training_dataset = MiniImageNet(
        './miniimagenet/',
        mini_batch_size=args.batch_size,
        ways=args.ways,
        shots=args.shots,
        query_size=args.query_size,
        device='cpu',
        data_fold='train'
    )
    
    test_dataset = MiniImageNet(
        './miniimagenet/',
        mini_batch_size=args.batch_size,
        ways=args.ways,
        shots=args.shots,
        query_size=args.query_size,
        device='cpu',
        data_fold='test'
    )
    
    return iter(
        DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            num_workers=20,
            collate_fn=loader_collate_fn,
            pin_memory=True,
            pin_memory_device='cuda',
            )
        ), iter(
            DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                collate_fn=loader_collate_fn,
                pin_memory=True,
                pin_memory_device='cuda',
                )
            )
    