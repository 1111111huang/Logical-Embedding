from abc import ABC, abstractmethod
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import csv
import copy
from utils import convert_to_one_hot


class PartialOrderDataset():
    def __init__(self):
        self.label = []
        self.input = []

    def produceTensor(self):
        return torch.FloatTensor(self.label), torch.FloatTensor(self.input)

    def produceStructuralLearningFile(self, name):
        try:
            os.system("mkdir " + name)
        except:
            pass
        facts_file = open(name + '/facts.txt', 'w+')
        relations_file = open(name + '/relations.txt', 'w+')
        test_file = open(name + '/test.txt', 'w+')
        entities_file = open(name + '/entities.txt', 'w+')
        train_file = open(name + '/train.txt', 'w+')
        valid_file = open(name + '/valid.txt', 'w+')
        input = []
        label = []
        l = len(self.input)
        ind = [i for i in range(l)]
        np.random.shuffle(ind)
        for i in ind:
            input += [self.input[i]]
            label += [self.label[i]]
        input = np.array(input)
        label = np.array(label)
        relations = list(set(label))
        entities = list(set(input.reshape((-1,))))
        valid_input, test_input, train_input, fact_input = np.split(input, [l // 10, l // 5, l // 2])
        valid_label, test_label, train_label, fact_label = np.split(label, [l // 10, l // 5, l // 2])
        print("data prepared")
        for i in range(len(fact_input)):
            facts_file.writelines(
                str(fact_input[i, 0]) + '\t' + str(fact_label[i]) + '\t' + str(fact_input[i, 1]) + '\n')
        for i in range(len(relations)):
            relations_file.writelines(str(relations[i]) + '\n')
        for i in range(len(entities)):
            entities_file.writelines(str(entities[i]) + '\n')
        for i in range(len(test_input)):
            test_file.writelines(
                str(test_input[i, 0]) + '\t' + str(test_label[i]) + '\t' + str(test_input[i, 1]) + '\n')
        for i in range(len(train_input)):
            train_file.writelines(
                str(train_input[i, 0]) + '\t' + str(train_label[i]) + '\t' + str(train_input[i, 1]) + '\n')
        for i in range(len(valid_input)):
            valid_file.writelines(
                str(valid_input[i, 0]) + '\t' + str(valid_label[i]) + '\t' + str(valid_input[i, 1]) + '\n')
        facts_file.close()
        relations_file.close()
        test_file.close()
        entities_file.close()
        train_file.close()
        valid_file.close()


class NumericalOrder(PartialOrderDataset):
    def __init__(self, low, upp, n, r, seed):
        random.seed(seed)
        label = [i for i in range(r)] * n
        input = []
        random.shuffle(label)

        if r == 2:
            for l in label:
                if l == 0:
                    a = random.randint(low + 1, upp)
                    b = random.randint(low, a - 1)  # a>b
                else:
                    a = random.randint(low, upp)
                    b = random.randint(a, upp)  # a<=b
                input += [[a, b]]
        if r == 3:
            for l in label:
                if l == 0:
                    a = random.randint(low + 1, upp)
                    b = random.randint(low, a - 1)  # a>b
                elif l == 2:
                    a = random.randint(low, upp - 1)
                    b = random.randint(a + 1, upp)  # a<b
                else:
                    a = random.randint(low, upp)
                    b = a  # a=b
                input += [[a, b]]
        self.input = input
        self.label = label


class DivisibilityOrder(PartialOrderDataset):
    def __init__(self, low, upp, n, r, seed):
        random.seed(seed)
        label = [i for i in ['divides', 'dividedBy', 'notRelated']] * n
        input = []
        random.shuffle(label)

        if r == 2:
            for l in label:
                a = random.randint(low, upp)
                b = random.randint(low, upp)
                while not (((l == 'notRelated') and (a % b == 0)) or ((l == 'divide') and (a % b != 0))):
                    a = random.randint(low, upp)
                    b = random.randint(low, upp)
                input += [[a, b]]
        if r == 3:
            for l in label:
                a = random.randint(low, upp)
                b = random.randint(low, upp)
                count = 0
                while not (((l == 'dividedBy') and (a % b == 0)) or ((l == 'divides') and (b % a == 0)) or (
                        (l == 'notRelated') and (b % a != 0) and (a % b != 0))):
                    a = random.randint(low, upp)
                    b = random.randint(low, upp)
                    count += 1
                # print(count)
                input += [[a, b]]
        self.input = input
        self.label = label
        print("initialized")


class Prime(DivisibilityOrder):
    def __init__(self, low, upp, n, r, seed):
        super().__init__(low, upp, n, r, seed)
        prime_list = []
        with open("primeNum.csv") as fd:
            rd = csv.reader(fd, delimiter="\t")
            for row in rd:
                for num in row:
                    prime_list += [int(num)]
        ind = np.random.randint(0, len(prime_list) - 1, size=n)
        for i in ind:
            self.input += [[prime_list[i], prime_list[i]]]
            self.label += ['prime']
        fd.close()


class Composite(DivisibilityOrder):
    def __init__(self, low, upp, n, r, seed):
        super().__init__(low, upp, n, r, seed)
        prime_list = []
        self.compinput = []
        self.complabel = []
        with open("primeNum.csv") as fd:
            rd = csv.reader(fd, delimiter="\t")
            for row in rd:
                for num in row:
                    prime_list += [int(num)]
        for i in range(n):
            num = 2
            while (num in prime_list):
                num = np.random.randint(low, upp)
            self.compinput += [[num, num]]
            self.complabel += ['composite']
            self.input += [[num, num]]
            self.label += ['same']
        fd.close()

    def produceStructuralLearningFile(self, name):
        try:
            os.system("mkdir " + name)
        except:
            pass
        facts_file = open(name + '/facts.txt', 'w+')
        relations_file = open(name + '/relations.txt', 'w+')
        test_file = open(name + '/test.txt', 'w+')
        entities_file = open(name + '/entities.txt', 'w+')
        train_file = open(name + '/train.txt', 'w+')
        valid_file = open(name + '/valid.txt', 'w+')
        input = []
        label = []
        l = len(self.input)
        ind = [i for i in range(l)]
        np.random.shuffle(ind)
        for i in ind:
            input += [self.input[i]]
            label += [self.label[i]]
        input = np.array(input)
        label = np.array(label)
        relations = list(set(label)) + ['composite']
        entities = list(set(input.reshape((-1,))))
        l2 = len(self.compinput)
        valid_input, test_input, train_input = np.split(self.compinput, [l2 // 5, 2 * l2 // 5])
        valid_label, test_label, train_label = np.split(self.complabel, [l2 // 5, 2 * l2 // 5])
        train_input_, fact_input = np.split(self.input, [l // 3])
        train_label_, fact_label = np.split(self.label, [l // 3])
        train_input = np.concatenate((train_input, train_input_), axis=0)
        train_label = np.concatenate((train_label, train_label_), axis=0)
        print("data prepared")
        for i in range(len(fact_input)):
            facts_file.writelines(
                str(fact_input[i, 0]) + '\t' + str(fact_label[i]) + '\t' + str(fact_input[i, 1]) + '\n')
        for i in range(len(relations)):
            relations_file.writelines(str(relations[i]) + '\n')
        for i in range(len(entities)):
            entities_file.writelines(str(entities[i]) + '\n')
        for i in range(len(test_input)):
            test_file.writelines(
                str(test_input[i, 0]) + '\t' + str(test_label[i]) + '\t' + str(test_input[i, 1]) + '\n')
        for i in range(len(train_input)):
            train_file.writelines(
                str(train_input[i, 0]) + '\t' + str(train_label[i]) + '\t' + str(train_input[i, 1]) + '\n')
        for i in range(len(valid_input)):
            valid_file.writelines(
                str(valid_input[i, 0]) + '\t' + str(valid_label[i]) + '\t' + str(valid_input[i, 1]) + '\n')
        facts_file.close()
        relations_file.close()
        test_file.close()
        entities_file.close()
        train_file.close()
        valid_file.close()


class Dataset(data.Dataset):
    def __init__(self, X):
        self.data = X
        self.label = torch.ones(X.shape[0], 1)
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y


class Data(object):
    def __init__(self, folder, seed, batch_size):
        np.random.seed(seed)
        self.seed = seed
        self.query_include_reverse = True
        self.batch_size = batch_size

        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")

        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)
        self.number_to_relation = {}
        for key, value in self.relation_to_number.items():
            self.number_to_relation[value] = key
            self.number_to_relation[value + self.num_relation] = "inv_" + key

        self.test_file = os.path.join(folder, "test.txt")
        self.train_file = os.path.join(folder, "train.txt")
        self.valid_file = os.path.join(folder, "valid.txt")
        # If clauses omitted, since facts.txt is always there
        self.facts_file = os.path.join(folder, "facts.txt")
        #  self.share_db = True

        self.test, self.num_test = self._parse_triplets(self.test_file)
        self.train, self.num_train = self._parse_triplets(self.train_file)
        # If clauses omitted, the valid file is always a file
        self.valid, self.num_valid = self._parse_triplets(self.valid_file)
        train_set = Dataset(self.train)
        valid_set = Dataset(self.valid)
        test_set = Dataset(self.test)
        # Data format: Tensor of shape(batch_size, 3, num_relation/num_entity)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)

        #  Omitted share_db
        self.facts, self.num_fact = self._parse_triplets(self.facts_file)
        self.matrix_db = self._db_to_matrix_db(self.facts)
        self.matrix_db_train = self.matrix_db
        self.matrix_db_test = self.matrix_db
        self.matrix_db_valid = self.matrix_db
        # Omitted while omitted the use_extra_facts property

        #  Omitted type_check if clauses
        #  self.domains = None
        #  self.num_operator = 2 * self.num_relation

        # Omitted: get rules for queries and their inverses appeared in train and test for getting attentions
        # self.query_for_rules = list(set(zip(*self.train)[0]) | set(zip(*self.test)[0]) | set(
        #    zip(*self._augment_with_reverse(self.train))[0]) | set(zip(*self._augment_with_reverse(self.test))[0]))
        # parser omitted

    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert (len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)

        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert (len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert (len(l) == 3)
                output.append((list(convert_to_one_hot(self.relation_to_number[l[1]], self.num_relation)) +
                               list(convert_to_one_hot(self.entity_to_number[l[0]], self.num_entity)) +
                               list(convert_to_one_hot(self.entity_to_number[l[2]], self.num_entity))))
        return torch.Tensor(output), len(output)

    def _db_to_matrix_db(self, db):
        matrix_db = np.zeros((self.num_relation, self.num_entity, self.num_entity))
        for i, fact in enumerate(db):
            temp = np.argwhere(fact.numpy()==1).reshape(-1)
            rel = int(temp[0])
            head = int(temp[1])-self.num_relation
            tail = int(temp[2])-self.num_relation-self.num_entity
            matrix_db[rel, head, tail] = 1
        return torch.Tensor(matrix_db)

    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0] + self.num_relation,
                                    triplet[2],
                                    triplet[1])]
        return augmented


# TODO: language query needs to be investigated
data = DivisibilityOrder(1, 10, 300, 3, 0)
data.produceStructuralLearningFile("comp10")
print(data.input)
print(data.label)
