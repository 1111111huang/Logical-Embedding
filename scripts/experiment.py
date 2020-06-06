import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
from utils import list_rules, print_rules, measure_acc, show_result
import torch.nn as nn
import torch.optim as optim
import torch


class Experiment:

    def __init__(self, option, model, data, epoch=0):
        self.epoch = epoch
        self.option = option
        self.model = model
        self.data = data
        self.optimizer = optim.Adam(self.model.parameters(), lr=option.learning_rate)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        # helpers
        self.msg_with_time = lambda msg: \
            "%s Time elapsed %0.2f hrs (%0.1f mins)" \
            % (msg, (time.time() - self.start) / 3600.,
               (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_train_loss = np.inf
        self.best_train_acc = 0.
        self.best_valid_loss = np.inf
        self.best_valid_acc = 0.
        self.best_test_loss = np.inf
        self.best_test_acc = 0.
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        self.early_stopped = False
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

    def train_one_epoch(self):
        tr_loss = 0
        tr_acc = 0
        l = 0
        for i, batch in enumerate(self.data.train_loader):
            inputs, labels = batch
            inputs = inputs.type(torch.float32)
            labels = labels.type(torch.float32)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            tr_loss += loss.item()
            tr_acc += measure_acc(outputs, labels)
            l += 1
        tr_loss /= l
        tr_acc /= l
        self.train_loss += [tr_loss]
        self.train_acc += [tr_acc]
        if self.best_train_loss > tr_loss:
            self.best_train_loss = tr_loss
        if self.best_train_acc < tr_acc:
            self.best_train_acc = tr_acc
        print("Epoch: ",self.epoch, "Loss: ",self.train_loss[-1])
        v_acc = 0
        v_loss = 0
        l = 0
        for i, batch in enumerate(self.data.valid_loader):
            inputs, labels = batch
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            outputs = self.model(inputs)
            v_acc += measure_acc(outputs, labels)
            v_loss += self.loss_func(outputs, labels)
            l += 1
        self.valid_loss += [v_loss / l]
        self.valid_acc += [v_acc / l]
        if self.best_valid_loss > v_loss:
            self.best_valid_loss = v_loss
        if self.best_valid_acc < v_acc:
            self.best_valid_acc = v_acc

        t_acc = 0
        t_loss = 0
        l = 0
        for i, batch in enumerate(self.data.test_loader):
            inputs, labels = batch
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            outputs = self.model(inputs)
            t_acc += measure_acc(outputs, labels)
            t_loss += self.loss_func(outputs, labels)
            l += 1
        self.test_loss += [t_loss / l]
        self.test_acc += [t_acc / l]
        if self.best_test_loss > t_loss:
            self.best_test_loss = t_loss
        if self.best_test_acc < t_acc:
            self.best_test_acc = t_acc

    def early_stop(self):
        return False
        loss_improve = self.best_valid_loss == self.valid_loss[-1]
        in_top_improve = self.best_valid_acc == self.valid_acc[-1]
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.train_one_epoch()
            self.epoch += 1
            torch.save({'epoch': self.epoch, 'model_state_dict': self. model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, './model.pt')
            print("Model saved at %s" % self.option.model_path)

            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))

        all_test_in_top = self.test_acc
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        show_result(self.train_acc,self.train_loss,self.valid_acc,self.valid_loss)
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)
        print(msg)
        self.log_file.write(msg + "\n")
        #pickle.dump([self.train_loss, self.valid_loss, self.test_loss],
        #            open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))

    def close_log_file(self):
        self.log_file.close()
