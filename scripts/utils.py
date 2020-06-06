import numpy as np
import sys
import os
import matplotlib.pyplot as pyplot
import numpy as np
import torch

def measure_acc(outputs, labels):
    acc=0
    outputs=torch.round(outputs)
    batch_size=outputs.shape[0]
    for i in range(batch_size):
        if outputs[i]==labels[i]:
            acc+=1
    return acc/batch_size

def convert_to_one_hot(ind, num):
    r=np.ones(num)
    r[ind]=1
    return r

def show_result(train_acc,train_loss,valid_acc,valid_loss):
    print("train acc: ", train_acc[-1], "train loss", train_loss[-1])
    print("validate acc: ", valid_acc[-1], "validate loss", valid_loss[-1])
    print(train_acc[-1], valid_acc[-1])
    print('Finished Training')
    pyplot.plot(np.array(train_loss), label="training set")
    pyplot.title("Loss vs Epochs")
    pyplot.ylabel("Loss")
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Epoch")
    pyplot.show()
    pyplot.plot(np.array(train_acc), label="training set")
    pyplot.title("Acc vs Epochs")
    pyplot.ylabel("Acc")
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Epoch")
    pyplot.show()
    pyplot.plot(np.array(valid_loss), label="validaiton set")
    pyplot.title("Loss vs Epochs")
    pyplot.ylabel("Loss")
    pyplot.legend(loc='upper right')
    pyplot.xlabel("Epoch")
    pyplot.show()
    pyplot.plot(np.array(valid_acc), label="validation set")
    pyplot.title("Acc vs Epochs")
    pyplot.ylabel("Acc")
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Epoch")
    pyplot.show()

    
def display_mat(mat):
    mat=mat.numpy()
    shape=mat.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            print(str(round(mat[i,j]))+'|',end=''),
        print()
        for j in range(shape[1]):
            print('------',end=''),
        print()


def list_rules(attn_ops, attn_mems, the):
    """
    Given attentions over operators and memories,
    enumerate all rules and compute the weights for each.

    Args:
        attn_ops: a list of num_step vectors,
                  each vector of length num_operator.
        attn_mems: a list of num_step vectors,
                   with length from 1 to num_step.
        the: early prune by keeping rules with weights > the

    Returns:
        a list of (rules, weight) tuples.
        rules is a list of operator ids.

    """

    num_step = len(attn_ops)
    paths = {t + 1: [] for t in range(num_step)}
    paths[0] = [([], 1.)]
    for t in range(num_step):
        for m, attn_mem in enumerate(attn_mems[t]):
            for p, w in paths[m]:
                paths[t + 1].append((p, w * attn_mem))
        if t < num_step - 1:
            new_paths = []
            for o, attn_op in enumerate(attn_ops[t]):
                for p, w in paths[t + 1]:
                    if w * attn_op > the:
                        new_paths.append((p + [o], w * attn_op))
            paths[t + 1] = new_paths
    this_the = min([the], max([w for (_, w) in paths[num_step]]))
    final_paths = filter(lambda x: x[1] >= this_the, paths[num_step])
    final_paths.sort(key=lambda x: x[1], reverse=True)

    return final_paths


def print_rules(q_id, rules, parser, query_is_language):
    """
    Print rules by replacing operator ids with operator names
    and formatting as logic rules.

    Args:
        q_id: the query id (the head)
        rules: a list of ([operator ids], weight) (the body)
        parser: a dictionary that convert q_id and operator_id to
                corresponding names

    Returns:
        a list of strings, each string is a printed rule
    """

    if len(rules) == 0:
        return []

    if not query_is_language:
        query = parser["query"][q_id]
    else:
        query = parser["query"](q_id)

    # assume rules are sorted from high to lows
    max_w = rules[0][1]
    # compute normalized weights also
    rules = [[rule[0], rule[1], rule[1] / max_w] for rule in rules]

    printed_rules = []
    for rule, w, w_normalized in rules:
        if len(rule) == 0:
            printed_rules.append(
                "%0.3f (%0.3f)\t%s(B, A) <-- equal(B, A)"
                % (w, w_normalized, query))
        else:
            lvars = [chr(i + 65) for i in range(1 + len(rule))]
            printed_rule = "%0.3f (%0.3f)\t%s(%c, %c) <-- " \
                           % (w, w_normalized, query, lvars[-1], lvars[0])
            for i, literal in enumerate(rule):
                if not query_is_language:
                    literal_name = parser["operator"][q_id][literal]
                else:
                    literal_name = parser["operator"][literal]
                printed_rule += "%s(%c, %c), " \
                                % (literal_name, lvars[i + 1], lvars[i])
            printed_rules.append(printed_rule[0: -2])

    return printed_rules

def convert_to_one_hot(ind,n):
    r=np.zeros(n)
    r[ind]=1
    return r