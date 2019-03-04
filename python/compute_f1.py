'''
compute real parsing f1 score given a report file (without pair ID) discarded
'''

import codecs
import os
import sys
import json
import sys
import numpy
from nltk import Tree
import math

from spinn.util.data import get_brackets
from spinn.data.nli.load_nli_data import LABEL_MAP

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']

eval_fname = "PATH/data/tree_distillation/prpn_distance/snli_distance/valid.json"
test_fname = "PATH/data/tree_distillation/prpn_distance/snli_distance/test.json"
mnli_dev = "PATH/data/nli/multinli_1.0/multinli_1.0_dev_matched.jsonl"

def tree2list(tree):
    '''
    copy from PRPN
    '''
    if isinstance(tree, Tree):
        if tree.label() in word_tags:
            return tree.leaves()[0]
        else:
            root = []
            for child in tree:
                c = tree2list(child)
                if c != []:
                    root.append(c)
            if len(root) > 1:
                return root
            elif len(root) == 1:
                return root[0]
    return []

def compute_f1(overlap, std_tree, model_tree):
    prec = float(len(overlap)) / (len(model_tree) + 1e-8)
    reca = float(len(overlap)) / (len(std_tree) + 1e-8)
    if len(std_tree) == 0:
        reca = 1.
        if len(model_tree) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree

def build_tree_by_definition(depth, sen):
    assert len(depth) + 1 == len(sen)

    if len(sen) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth) + 1
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree_by_definition(depth[:idx_max - 1], sen[:idx_max])
            parse_tree.append(tree0)
        if len(sen[idx_max:]) > 0:
            tree1 = build_tree_by_definition(depth[idx_max :], sen[idx_max :])
            parse_tree.append(tree1)
    return parse_tree


def get_balanced_tree(max_length):
    l = range(max_length)

    def get_btree(l):
        if len(l) < 3:
            return l
        else:
            # split = int(math.ceil(len(l)/2.0))
            split = len(l)/2
            return [get_btree(l[:split]), get_btree(l[split:])]
    return get_btree(l)


def verify_f1(path):
    f1_list = []
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            t1 = Tree.fromstring(loaded_example['sentence1_parse'])
            l1 = len(t1.leaves())
            t1 = tree2list(t1)
            t2 = Tree.fromstring(loaded_example['sentence2_parse'])
            l2 = len(t2.leaves())
            t2 = tree2list(t2)
            # print t1
            # print l1
            # print t2
            # print l2

            bt1 = get_balanced_tree(l1)
            bt2 = get_balanced_tree(l2)
            # print bt1
            # print bt2

            print t1
            t1 = get_brackets(t1)[0]
            print t1
            sys.exit(0)

            t2 = get_brackets(t2)[0]
            bt1 = get_brackets(bt1)[0]
            bt2 = get_brackets(bt2)[0]

            # t1.add((0,l1))
            # bt1.add((0,l1))
            # t2.add((0,l2))
            # bt2.add((0,l2))

            # print t1
            # print t2
            # print bt1
            # print bt2

            f1 = compute_f1(t1 & bt1, t1, bt1)
            f1_list.append(f1)
            f1 = compute_f1(t2 & bt2, t2, bt2)
            f1_list.append(f1)

    return sum(f1_list) / len(f1_list), len(f1_list)
            

def compute_f1_baseline(path):
    '''
    for RL
    '''

    rb_f1_list = []
    lb_f1_list = []
    prpn_f1_list = []
    prpn_f1_df_list = []
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                    # 158 here
                continue

            prpn_gates1 = loaded_example['sentence1_prpn_gates']
            prpn_gates2 = loaded_example['sentence2_prpn_gates']
            prpn_df_tree1 = get_brackets(build_tree_by_definition(prpn_gates1[1:], loaded_example['sentence1']))[0]
            prpn_df_tree2 = get_brackets(build_tree_by_definition(prpn_gates2[1:], loaded_example['sentence2']))[0]

            std_tree1 = get_brackets(loaded_example['sentence1_binary_parse'])[0]
            prpn_tree1 = get_brackets(loaded_example['sentence1_prpn_binary_parse'])[0]
            std_tree2 = get_brackets(loaded_example['sentence2_binary_parse'])[0]
            prpn_tree2 = get_brackets(loaded_example['sentence2_prpn_binary_parse'])[0]

            len1 = len(loaded_example['sentence1'])
            if len1 < 3:
                lb_tree1 = set()
                rb_tree1 = set()
            else:
                lb_tree1 = {(0, i) for i in range(2,len1-1)}
                rb_tree1 = {(i, len1) for i in range(1,len1-2)}
            len2 = len(loaded_example['sentence2'])
            if len2 < 3:
                lb_tree2 = set()
                rb_tree2 = set()
            else:
                lb_tree2 = {(0, i) for i in range(2,len2-1)}
                rb_tree2 = {(i, len2) for i in range(1,len2-2)}

            rb_f1_list.append(compute_f1(rb_tree1 & std_tree1, std_tree1, rb_tree1))
            rb_f1_list.append(compute_f1(rb_tree2 & std_tree2, std_tree2, rb_tree2))
            lb_f1_list.append(compute_f1(lb_tree1 & std_tree1, std_tree1, lb_tree1))
            lb_f1_list.append(compute_f1(lb_tree2 & std_tree2, std_tree2, lb_tree2))
            prpn_f1_list.append(compute_f1(prpn_tree1 & std_tree1, std_tree1, prpn_tree1))
            prpn_f1_list.append(compute_f1(prpn_tree2 & std_tree2, std_tree2, prpn_tree2))
            prpn_f1_df_list.append(compute_f1(prpn_df_tree1 & std_tree1, std_tree1, prpn_df_tree1))
            prpn_f1_df_list.append(compute_f1(prpn_df_tree2 & std_tree2, std_tree2, prpn_df_tree2))

    rb_f1 = sum(rb_f1_list) / len(rb_f1_list)
    lb_f1 = sum(lb_f1_list) / len(lb_f1_list)
    prpn_f1 = sum(prpn_f1_list) / len(prpn_f1_list)
    prpn_f1_df = sum(prpn_f1_df_list) / len(prpn_f1_df_list)

    return rb_f1, lb_f1, prpn_f1, prpn_f1_df

# print 'eval set: right-branching {:.3f} left-branching {:.3f} prpn {:.3f} prpn_df {:.3f}'\
#     .format(*compute_f1_baseline(eval_fname))
# print 'test set: right-branching {:.3f} left-branching {:.3f} prpn {:.3f} prpn_df {:.3f}'\
#     .format(*compute_f1_baseline(test_fname))

print verify_f1(mnli_dev)