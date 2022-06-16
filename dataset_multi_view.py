import os
import glob
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MultiViewDataset(Dataset):
    def __init__(self, classes, num_classes, data_root, mode, max_num_views):
        super(MultiViewDataset, self).__init__()
 
        self.classes = classes
        self.num_classes = num_classes
        self.data_root = data_root
        self.mode = mode
        self.max_num_views = max_num_views

        self.file_path = []

        for class_name in self.classes:
            if self.mode == 'train':
                mode_path = os.path.join(self.data_root, mode, class_name)
                for set_file in sorted(os.listdir(mode_path)):
                    files = sorted(glob.glob(os.path.join(mode_path, set_file, '*.png')))
                    self.file_path.append(files)
            else:
                class_list = self.data_root + mode + '_list/' + mode + '_' + class_name + '_100.pkl'
                open_file = open(class_list, 'rb')
                list_path = pickle.load(open_file)
                open_file.close()
                self.file_path += list_path

        if self.mode == 'train':
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

        for i in range(0, len(self.file_path[index])):
            if self.mode == 'train':
                image = Image.open(self.file_path[index][i]).convert('RGB')
            else:
                image = Image.open(os.path.join(self.data_root, self.mode, self.file_path[index][i])).convert('RGB')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        for i in range(0, self.max_num_views - len(self.file_path[index])):
            images.append(torch.zeros_like(images[0]))
 
        return label, torch.stack(images), len(self.file_path[index])

