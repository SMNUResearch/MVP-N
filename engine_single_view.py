import os
import math
import time
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class SingleViewEngine(object):
    def __init__(self, model, train_data, valid_data, num_classes, optimizer, sv_type, criterion, weight_path, output_path, device, single_view=True):
        super(SingleViewEngine, self).__init__()

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.sv_type = sv_type
        self.criterion = criterion
        self.weight_path = weight_path
        self.output_path = output_path
        self.device = device
        self.single_view = single_view

        self.best_accuracy = 0
        self.start_epoch = 0

    def save_model_weights(self, epoch, overall_accuracy):
        best_weight = os.path.join(self.weight_path, self.sv_type + '.pt')
        if overall_accuracy >= self.best_accuracy:
            self.best_accuracy = overall_accuracy
            print('Save Weight!')
            torch.save(self.model.state_dict(), best_weight, _use_new_zipfile_serialization=False)

    def train_base(self, epochs, data_length, save_outputs=False):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            total_outputs = torch.zeros(data_length, self.num_classes).to(self.device)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if save_outputs == True:
                    total_outputs[image_index] = outputs.clone().detach().to(self.device)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

            # save outputs (teacher)
            if save_outputs == True:
                torch.save(total_outputs, os.path.join(self.output_path, self.sv_type + '.pt'))

    def train_bs(self, epochs):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, epoch)

                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_kd(self, epochs, outputs_teacher):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, torch.load(outputs_teacher)[image_index].to(self.device))
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))
            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_db(self, epochs):
        self.model.train()
        previous_time = time.time()
        max_loss_epoch = 0
        min_loss_epoch = 0

        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            loss_overall = torch.tensor([])
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss_ce, loss = self.criterion(outputs, targets, epoch, max_loss_epoch, min_loss_epoch)
                loss_overall = torch.cat((loss_overall.cpu().data, loss_ce.cpu().data), dim=-1)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            max_loss_epoch, min_loss_epoch = self.criterion.update(loss_overall)

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_sat(self, epochs):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, epoch, image_index)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_lrt_plc(self, epochs):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, epoch, image_index)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.criterion.update(epoch)

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

    def train_seal(self, epochs, data_length, seal_time, seal_targets):
        self.model.train()
        previous_time = time.time()
        total_outputs = torch.zeros(data_length, self.num_classes).to(self.device)
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                confidence = F.softmax(outputs.detach(), dim=1).to(self.device)
                total_outputs[image_index] += confidence

                if seal_time == 0:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs, torch.load(seal_targets)[image_index])

                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            # get remaining time
            current_time = time.time()
            time_left = datetime.timedelta(seconds=(current_time - previous_time)*(epochs - epoch - 1))

            previous_time = current_time
            print('Remaining Time:', time_left)

        total_outputs = total_outputs / epochs
        torch.save(total_outputs, os.path.join(self.output_path, self.sv_type + str(seal_time) + '.pt'))

    def train_ols(self, epochs):
        self.model.train()
        previous_time = time.time()
        for epoch in range(self.start_epoch, epochs):
            total_loss = 0
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for index, (label, image, path, image_index) in enumerate(self.train_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.criterion.update()
            script = ('Epoch:[ %d | %d ]    Loss: %.4f    ') % (epoch + 1, epochs, total_loss)
            print(script)

            if self.single_view == True:
                # evaluation
                with torch.no_grad():
                    overall_accuracy = self.valid()

                # save best model
                self.save_model_weights(epoch, overall_accuracy)

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

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
            for index, (label, image, path, image_index) in enumerate(self.valid_data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                outputs = self.model(inputs)
                prediction = torch.max(outputs, 1)[1]
                transform_targets = torch.max(targets, 1)[1]
                results = (prediction == transform_targets)
                correct_points = torch.sum(results.long())
                all_correct_points += correct_points
                all_points += results.size()[0]

        overall_accuracy = (all_correct_points.float() / all_points).cpu().data.numpy()
        print('SVA:', '%.2f' % (100 * overall_accuracy))

        self.model.train()

        return overall_accuracy

    def test(self, data, T):
        all_correct_points = 0
        all_points = 0
        info_correct_points = 0
        info_points = 0
        wrong_class = np.zeros(self.num_classes)
        samples_class = np.zeros(self.num_classes)

        bin_confidence = np.zeros(10)
        bin_count = np.zeros(10)
        bin_count_wrong = np.zeros(10)

        bin_info_correct_confidence = np.zeros(10)
        bin_info_wrong_confidence = np.zeros(10)
        bin_info_count_correct = np.zeros(10)
        bin_info_count_wrong = np.zeros(10)

        wrong_difference = 0
        wrong_uninfo_count = 0

        self.model.eval()

        with torch.no_grad():
            for index, (label, image, path, image_index) in enumerate(data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                outputs = self.model(inputs)
                prediction = torch.max(outputs, 1)[1]
                transform_targets = torch.max(targets, 1)[1]
                results = (prediction == transform_targets)
                confidence = F.softmax(outputs / T, dim=1)
                for i in range(results.size()[0]):
                    image_name = (path[i].strip().split('/')[-1]).strip().split('.')[0]
                    info_mark = int(image_name.strip().split('_')[-1])
                    bin_index = math.ceil(torch.max(confidence[i]) * 10)
                    if info_mark == 1:
                        info_points += 1
                        if not bool(results[i].cpu().data.numpy()):
                            bin_info_count_wrong[bin_index - 1] += 1
                            bin_info_wrong_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()
                        else:
                            info_correct_points += 1
                            bin_info_count_correct[bin_index - 1] += 1
                            bin_info_correct_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()
                    else:
                        if not bool(results[i].cpu().data.numpy()):
                            difference = torch.max(confidence[i]).cpu().data.numpy() - confidence[i][transform_targets.cpu().data.numpy().astype('int')[i]].cpu().data.numpy()
                            wrong_difference += difference
                            wrong_uninfo_count += 1

                    samples_class[transform_targets.cpu().data.numpy().astype('int')[i]] += 1
                    bin_count[bin_index - 1] += 1
                    bin_confidence[bin_index - 1] += torch.max(confidence[i]).cpu().data.numpy()
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[transform_targets.cpu().data.numpy().astype('int')[i]] += 1
                        bin_count_wrong[bin_index - 1] += 1

                correct_points = torch.sum(results.long())
                all_correct_points += correct_points
                all_points += results.size()[0]

        mean_class_accuracy = np.mean((samples_class - wrong_class) / samples_class)
        overall_accuracy = (all_correct_points.float() / all_points).cpu().data.numpy()
        info_overall_accuracy = (1.0 * info_correct_points / info_points)
        ECE = np.sum(np.abs(bin_count - bin_count_wrong - bin_confidence)) / all_points
        print('SVA:', '%.2f' % (100 * overall_accuracy))
        print('SVAI:', '%.2f' % (100 * info_overall_accuracy))
        print('MCCI:', '%.4f' % (bin_info_correct_confidence.sum() / bin_info_count_correct.sum()))
        print('MCWI:', '%.4f' % (bin_info_wrong_confidence.sum() / bin_info_count_wrong.sum()))
        print('MCDU:', '%.4f' % (wrong_difference / wrong_uninfo_count))

    def score_fusion(self, data, T):
        all_correct_points = 0
        all_points = 0
        view_num_count = np.zeros(5)
        view_num_correct_count = np.zeros(5)

        self.model.eval()

        with torch.no_grad():
            for index, (label, image, num_views) in enumerate(data):
                inputs = Variable(image).to(self.device)
                targets = Variable(label).to(self.device)
                B, V, C, H, W = inputs.shape
                transform_targets = torch.max(targets, 1)[1]
                for i in range(0, B):
                    view_set = inputs[i]
                    view_num = num_views[i].data.numpy().astype('int')
                    views = view_set[0:view_num]
                    outputs = self.model(views)
                    prediction = torch.max(outputs, 1)[1]
                    confidence = F.softmax(outputs / T, dim=1)
                    mean_rule = torch.sum(confidence, dim=0)
                    all_points += 1
                    view_num_count[view_num - 2] += 1
                    if torch.argmax(mean_rule) == transform_targets[i]:
                        all_correct_points += 1
                        view_num_correct_count[view_num - 2] += 1

        view_num_accuracy = view_num_correct_count / view_num_count
        print('MVA:', '%.2f' % (100 * all_correct_points / all_points))
        print('MVA (# of Views):', '%.2f' % (100 * view_num_accuracy[0]), '%.2f' % (100 * view_num_accuracy[1]), '%.2f' % (100 * view_num_accuracy[2]), '%.2f' % (100 * view_num_accuracy[3]), '%.2f' % (100 * view_num_accuracy[4]))

