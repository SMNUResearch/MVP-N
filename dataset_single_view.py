import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SingleViewDataset(Dataset):
    def __init__(self, classes, groups, num_classes, data_root, mode, sv_type, use_train=False):
        super(SingleViewDataset, self).__init__()

        self.classes = classes
        self.groups = groups
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.sv_type = sv_type
        self.use_train = use_train

        self.file_path = []

        for class_name in self.classes:
            mode_path = os.path.join(self.data_root, self.mode, class_name)
            for set_file in sorted(os.listdir(mode_path)):
                files = sorted(glob.glob(os.path.join(mode_path, set_file, '*.png')))
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

    def find_group(self, groups, class_name):
        for i, x in enumerate(groups):
            if class_name in x:
                return x

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        path = self.file_path[index]
        class_name = path.split('/')[-3]
        class_id = self.classes.index(class_name)
        image = Image.open(self.file_path[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.sv_type == 'HPIQ':
            label = torch.zeros(self.num_classes)
            if self.use_train:
                image_file = path.split('/')[-1]
                image_name = image_file.strip().split('.')[0]
                info_mark = int(image_name.strip().split('_')[-1])
                if info_mark == 1:
                    label[class_id] = 1.0
                else:
                    group = self.find_group(self.groups, class_name)
                    for i, x in enumerate(group):
                        label[self.classes.index(x)] = 1.0 / len(group)
            else:
                label[class_id] = 1.0
        elif self.sv_type == 'HS':
            if self.use_train:
                image_file = path.split('/')[-1]
                image_name = image_file.strip().split('.')[0]
                info_mark = int(image_name.strip().split('_')[-1])
                label = torch.zeros(self.num_classes * 2)
                label[class_id * 2 + info_mark] = 1.0
            else:
                label = torch.zeros(self.num_classes)
                label[class_id] = 1.0
        else:
            label = torch.zeros(self.num_classes)
            label[class_id] = 1.0

        return label, image, path, index

