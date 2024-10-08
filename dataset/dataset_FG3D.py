import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FG3DSingleViewDataset(Dataset):
    def __init__(self, data_root, category, mode, use_train=False):
        super(FG3DSingleViewDataset, self).__init__()

        self.data_root = data_root + category + '_view/'
        self.mode_file = data_root + category + '_view_' + mode + '.txt'
        self.use_train = use_train
        sets = sorted(os.listdir(self.data_root))

        self.file_path = []
        self.classes = []

        image_list = open(self.mode_file, 'r').readlines()
        for i in range(0, len(image_list)):
            image_name = image_list[i].split('\n')[0]
            class_name = image_name.split('_')[0]
            if class_name not in self.classes:
                self.classes.append(class_name)

            self.file_path.append(self.data_root + image_name + '.png')

        self.num_classes = len(self.classes)

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
        image = Image.open(self.file_path[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        path = self.file_path[index]
        class_name = (path.split('/')[-1]).split('_')[0]
        class_id = self.classes.index(class_name)
        label = torch.zeros(self.num_classes)
        label[class_id] = 1.0

        return label, image, path, index

class FG3DMultiViewDataset(Dataset):
    def __init__(self, data_root, category, mode, num_views, use_train=False):
        super(FG3DMultiViewDataset, self).__init__()

        self.data_root = data_root + category + '_view/'
        self.mode_file = data_root + category + '_view_' + mode + '.txt'
        self.num_views = num_views
        self.use_train = use_train
        sets = sorted(os.listdir(data_root))

        self.file_path = []
        self.classes = []

        image_list = open(self.mode_file, 'r').readlines()
        for i in range(0, len(image_list)):
            image_name = image_list[i].split('\n')[0]
            class_name = image_name.split('_')[0]
            if class_name not in self.classes:
                self.classes.append(class_name)

            self.file_path.append(self.data_root + image_name + '.png')

        self.num_classes = len(self.classes)

        self.set_path = []
        for i in range(0, int(len(image_list) / self.num_views)):
            self.set_path.append(self.file_path[(i * self.num_views):((i + 1) * self.num_views)])

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
        return len(self.set_path)

    def __getitem__(self, index):
        class_name = (self.set_path[index][0].split('/')[-1]).split('_')[0]
        class_id = self.classes.index(class_name)
        label = torch.zeros(self.num_classes)
        label[class_id] = 1.0
        images = []
        marks = torch.zeros(self.num_views)

        for i in range(0, len(self.set_path[index])):
            image = Image.open(self.set_path[index][i]).convert('RGB')
            if self.transform:
                image = self.transform(image)

            images.append(image)

        return label, torch.stack(images), len(self.set_path[index]), marks

