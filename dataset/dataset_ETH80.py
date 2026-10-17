import os
import glob

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SingleViewDatasetETH80(Dataset):
    def __init__(self, classes, num_classes, data_root, mode, use_train=False):
        super().__init__()

        self.classes = classes
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.use_train = use_train

        self.file_path = []

        for class_name in self.classes:
            class_path = os.path.join(self.data_root, class_name)
            for set_file in sorted(os.listdir(class_path)):
                set_index = int(set_file.split('_')[1])
                # can freely define a custom train/test split here;
                if self.mode == 'train':
                    split_index = [1, 3, 5, 7, 9]
                else:
                    split_index = [2, 4, 6, 8, 10]

                if set_index in split_index:
                    files = sorted(glob.glob(os.path.join(class_path, set_file, '*.png')))
                    self.file_path.extend(files)

        if self.use_train:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        path = self.file_path[index]
        class_name = path.split('/')[-3]
        class_id = self.classes.index(class_name)
        image = Image.open(self.file_path[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.zeros(self.num_classes)
        label[class_id] = 1.0

        return label, image, path, index

class MultiViewDatasetETH80(Dataset):
    def __init__(self, classes, num_classes, data_root, mode, max_num_views, use_train=False):
        super().__init__()

        self.classes = classes
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.max_num_views = max_num_views
        self.use_train = use_train

        self.file_path = []

        for class_name in self.classes:
            class_path = os.path.join(self.data_root, class_name)
            for set_file in sorted(os.listdir(class_path)):
                set_index = int(set_file.split('_')[1])
                # can freely define a custom train/test split here;
                if self.mode == 'train':
                    split_index = [1, 3, 5, 7, 9]
                else:
                    split_index = [2, 4, 6, 8, 10]

                if set_index in split_index:
                    files = sorted(glob.glob(os.path.join(class_path, set_file, '*.png')))
                    self.file_path.append(files)

        if self.use_train:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        class_name = self.file_path[index][0].split('/')[-3]
        class_id = self.classes.index(class_name)
        label = torch.zeros(self.num_classes)
        label[class_id] = 1.0
        images = []
        marks = torch.zeros(self.max_num_views)

        for i in range(0, len(self.file_path[index])):
            image = Image.open(self.file_path[index][i]).convert('RGB')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        for i in range(0, self.max_num_views - len(self.file_path[index])):
            images.append(torch.zeros_like(images[0]))

        return label, torch.stack(images), len(self.file_path[index]), marks

