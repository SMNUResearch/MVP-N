import os
import math
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils as tool

class MultiViewEngine(object):
    def __init__(self, model, train_data, valid_data, num_classes, optimizer, scheduler, criterion, weight_path, device, mv_type):
        super(MultiViewEngine, self).__init__()

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.weight_path = weight_path
        self.device = device
        self.mv_type = mv_type
        self.best_accuracy = 0
        self.start_epoch = 0

    def save_model_weights(self, epoch, overall_accuracy):
        best_weight = os.path.join(self.weight_path, self.mv_type + '.pt')
        if overall_accuracy >= self.best_accuracy:
            self.best_accuracy = overall_accuracy
            print('Save Weight!')
            torch.save(self.model.state_dict(), best_weight, _use_new_zipfile_serialization=False)

    def train_base(self, epochs):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            for index, (label, image, num_views) in enumerate(self.train_data):
                self.scheduler.step(len(self.train_data)*epoch + index)
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features = self.model(B, V, num_views, inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            # evaluation
            with torch.no_grad():
                overall_accuracy = self.valid()

            # save best model
            self.save_model_weights(epoch, overall_accuracy)

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))
            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_cvr(self, epochs, vert, weight_factor, norm_eps):
        self.model.train()
        previous_time = time.time()
        vert = torch.Tensor(vert).to(self.device)
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            for index, (label, image, num_views) in enumerate(self.train_data):
                self.scheduler.step(len(self.train_data)*epoch + index)
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features, pos = self.model(B, V, num_views, inputs)
                pos_loss = torch.norm(tool.normalize(vert, norm_eps) - tool.normalize(pos, norm_eps), p=2, dim=-1).mean()
                loss = self.criterion(outputs, targets) + weight_factor * pos_loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            # evaluation
            with torch.no_grad():
                overall_accuracy = self.valid()

            # save best model
            self.save_model_weights(epoch, overall_accuracy)

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))
            previous_time = current_time
            print('Remaining Time:', time_left)

    def valid(self):
        all_correct_points = 0
        all_points = 0
        self.model.eval()

        with torch.no_grad():
            for index, (label, image, num_views) in enumerate(self.valid_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features = self.model(B, V, num_views, inputs)
                prediction = torch.max(outputs, 1)[1]
                transform_targets = torch.max(targets, 1)[1]
                results = (prediction == transform_targets)
                correct_points = torch.sum(results.long())
                all_correct_points += correct_points
                all_points += results.size()[0]

        overall_accuracy = (all_correct_points.float() / all_points).cpu().data.numpy()
        print('MVA:', '%.2f' % (100 * overall_accuracy))

        self.model.train()

        return overall_accuracy

    def test(self, data, T):
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(self.num_classes)
        samples_class = np.zeros(self.num_classes)
        view_num_count = np.zeros(5)
        view_num_correct_count = np.zeros(5)

        bin_confidence = np.zeros(10)
        bin_correct_confidence = np.zeros(10)
        bin_wrong_confidence = np.zeros(10)
        bin_count = np.zeros(10)
        bin_correct_count = np.zeros(10)
        bin_wrong_count = np.zeros(10)

        self.model.eval()

        with torch.no_grad():
            for index, (label, image, num_views) in enumerate(data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features = self.model(B, V, num_views, inputs)
                prediction = torch.max(outputs, 1)[1]
                transform_targets = torch.max(targets, 1)[1]
                results = (prediction == transform_targets)
                confidence = F.softmax(outputs / T, dim=1)
                for i in range(results.size()[0]):
                    bin_index = math.ceil(torch.max(confidence[i]) * 10)
                    bin_count[bin_index - 1] += 1
                    bin_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()
                    samples_class[transform_targets.cpu().data.numpy().astype('int')[i]] += 1
                    view_num_count[num_views[i] - 2] += 1
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[transform_targets.cpu().data.numpy().astype('int')[i]] += 1
                        bin_wrong_count[bin_index - 1] += 1
                        bin_wrong_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()
                    else:
                        view_num_correct_count[num_views[i] - 2] += 1
                        bin_correct_count[bin_index - 1] += 1
                        bin_correct_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()

                correct_points = torch.sum(results.long())
                all_correct_points += correct_points
                all_points += results.size()[0]

        ECE = np.sum(np.abs(bin_correct_count - bin_confidence)) / all_points
        mean_class_accuracy = np.mean((samples_class - wrong_class) / samples_class)
        overall_accuracy = (all_correct_points.float() / all_points).cpu().data.numpy()
        view_num_accuracy = view_num_correct_count / view_num_count
        print('MVA:', '%.2f' % (100 * overall_accuracy))
        print('MCC:', '%.4f' % (bin_correct_confidence.sum() / bin_correct_count.sum()))
        print('MCW:', '%.4f' % (bin_wrong_confidence.sum() / bin_wrong_count.sum()))
        print('MVA (# of Views):', '%.2f' % (100 * view_num_accuracy[0]), '%.2f' % (100 * view_num_accuracy[1]), '%.2f' % (100 * view_num_accuracy[2]), '%.2f' % (100 * view_num_accuracy[3]), '%.2f' % (100 * view_num_accuracy[4]))

    def confusion_matrix(self, data):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.model.eval()

        with torch.no_grad():
            for index, (label, image, num_views) in enumerate(data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                B, V, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                outputs, features = self.model(B, V, num_views, inputs)
                prediction = torch.max(outputs, 1)[1]
                transform_targets = torch.max(targets, 1)[1]
                for i in range(outputs.size()[0]):
                    confusion_matrix[transform_targets.cpu().data.numpy().astype('int')[i]][prediction.cpu().data.numpy().astype('int')[i]] += 1

        return confusion_matrix

