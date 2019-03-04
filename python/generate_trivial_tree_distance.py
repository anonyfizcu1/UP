'''
generate trivial tree (left-branching, right-branching, balanced tree) data 
as initial policy for gumbel training for ALLNLI dataset
'''
'''
compute f1 score
'''

import codecs
import os
import sys
import json
import sys
import numpy
from nltk import Tree
import math
import re


from spinn.util.data import get_brackets
from spinn.data.nli.load_nli_data import LABEL_MAP

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
all_tags = word_tags + punctuation_tags

allnli_dir = 'PATH/data/nli/all_nli'
allnli_train_fname = 'multinli_1.0_snli_1.0_train_combined.jsonl'
allnli_valid_fname = 'multinli_1.0_dev_matched.jsonl'

target_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'

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

def filter_words(tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                # if tag == 'CD':
                #     w = 'N'
                words.append(w)
        return words

def compute_f1(overlap, std_tree, model_tree):
    prec = float(len(overlap)) / (len(model_tree) + 1e-8)
    reca = float(len(overlap)) / (len(std_tree) + 1e-8)
    if len(std_tree) == 0:
        reca = 1.
        if len(model_tree) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

def get_balanced_tree(words):
    if len(words) < 3:
        return words
    else:
        # split = int(math.ceil(len(l)/2.0))
        split = len(words)/2
        return [get_balanced_tree(words[:split]), get_balanced_tree(words[split:])]

def get_left_branching_tree(words):
    if len(words) < 3:
        return words
    else:
        return [get_left_branching_tree(words[:-1]), words[-1]]

def get_right_branching_tree(words):
    if len(words) < 3:
        return words
    else:
        return [words[0], get_right_branching_tree(words[1:])]

def generate_trivial_tree_dataset(read_file_path, write_file_path, trivial_tree='balanced'):
    if trivial_tree == 'balanced':
        get_trivial_tree = get_balanced_tree
    elif trivial_tree == 'left_branching':
        get_trivial_tree = get_left_branching_tree
    elif trivial_tree == 'right_branching':
        get_trivial_tree = get_right_branching_tree
    else:
        raise ValueError('invalid trivial tree form!')
    print '****** generating {} tree ******'.format(trivial_tree)

    f1_list = []

    fw = open(write_file_path, 'w')

    with codecs.open(read_file_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            write_example = {}

            write_example['gold_label'] = loaded_example['gold_label']
            if 'genre' in loaded_example:
                write_example['genre'] = loaded_example['genre']
            if 'promptID' in loaded_example:
                write_example['promptID'] = loaded_example['promptID']

            t1 = Tree.fromstring(loaded_example['sentence1_parse'])
            t2 = Tree.fromstring(loaded_example['sentence2_parse'])
            words1 = filter_words(t1)
            words2 = filter_words(t2)

            trivial_t1 = get_trivial_tree(words1)
            trivial_t2 = get_trivial_tree(words2)

            write_example['sentence1_prpn_binary_parse'] = trivial_t1
            write_example['sentence2_prpn_binary_parse'] = trivial_t2
            write_example['sentence1_binary_parse'] = tree2list(t1)
            write_example['sentence2_binary_parse'] = tree2list(t2)
            write_example['sentence1'] = words1
            write_example['sentence2'] = words2

            fw.write(json.dumps(write_example))
            fw.write('\n')

            t1_brackets, l1 = get_brackets(tree2list(t1))
            t2_brackets, l2 = get_brackets(tree2list(t2))
            trivial_t1_brackets, trivial_l1 = get_brackets(trivial_t1)
            trivial_t2_brackets, trivial_l2 = get_brackets(trivial_t2)
            assert l1 == len(words1)
            assert l2 == len(words2)
            assert l1 == trivial_l1
            assert l2 == trivial_l2

            t1_brackets.add((0,l1))
            trivial_t1_brackets.add((0,l1))
            t2_brackets.add((0,l2))
            trivial_t2_brackets.add((0,l2))

            f1 = compute_f1(t1_brackets & trivial_t1_brackets, t1_brackets, trivial_t1_brackets)
            f1_list.append(f1)
            f1 = compute_f1(t2_brackets & trivial_t2_brackets, t2_brackets, trivial_t2_brackets)
            f1_list.append(f1)
    fw.close()

    return sum(f1_list) / len(f1_list), len(f1_list)

def generate_trivial_tree_dataset_debug(read_file_path, write_file_path, trivial_tree='balanced'):
    if trivial_tree == 'balanced':
        get_trivial_tree = get_balanced_tree
    elif trivial_tree == 'left_branching':
        get_trivial_tree = get_left_branching_tree
    elif trivial_tree == 'right_branching':
        get_trivial_tree = get_right_branching_tree
    else:
        raise ValueError('invalid trivial tree form!')
    print '****** generating {} tree ******'.format(trivial_tree)

    f1_list = []

    with codecs.open(read_file_path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            write_example = {}

            write_example['gold_label'] = loaded_example['gold_label']
            if 'genre' in loaded_example:
                write_example['genre'] = loaded_example['genre']
            if 'promptID' in loaded_example:
                write_example['promptID'] = loaded_example['promptID']

            t1 = Tree.fromstring(loaded_example['sentence1_parse'])
            t2 = Tree.fromstring(loaded_example['sentence2_parse'])
            words1 = filter_words(t1)
            words2 = filter_words(t2)

            if len(words1) < 1 or len(words2) < 1:
                continue

            trivial_t1 = get_trivial_tree(words1)
            trivial_t2 = get_trivial_tree(words2)

            write_example['sentence1_prpn_binary_parse'] = trivial_t1
            write_example['sentence2_prpn_binary_parse'] = trivial_t2
            write_example['sentence1_binary_parse'] = tree2list(t1)
            write_example['sentence2_binary_parse'] = tree2list(t2)
            write_example['sentence1'] = words1
            write_example['sentence2'] = words2

            t1_brackets, l1 = get_brackets(tree2list(t1))
            t2_brackets, l2 = get_brackets(tree2list(t2))
            trivial_t1_brackets, trivial_l1 = get_brackets(trivial_t1)
            trivial_t2_brackets, trivial_l2 = get_brackets(trivial_t2)
            assert l1 == len(words1)
            assert l2 == len(words2)
            assert l1 == trivial_l1
            assert l2 == trivial_l2

            t1_brackets.add((0,l1))
            trivial_t1_brackets.add((0,l1))
            t2_brackets.add((0,l2))
            trivial_t2_brackets.add((0,l2))

            f1 = compute_f1(t1_brackets & trivial_t1_brackets, t1_brackets, trivial_t1_brackets)
            f1_list.append(f1)
            f1 = compute_f1(t2_brackets & trivial_t2_brackets, t2_brackets, trivial_t2_brackets)
            f1_list.append(f1)

    return sum(f1_list) / len(f1_list), len(f1_list)


# for tree in ('balanced', 'left_branching', 'right_branching'):
for tree in ('right_branching', 'left_branching', 'right_branching'):
    # print 'train set'
    # print generate_trivial_tree_dataset_debug(
    #     os.path.join(allnli_dir, allnli_train_fname), 
    #     os.path.join(target_dir, 'allnli_train_wo-p_{}.json'.format(tree)), 
    #     trivial_tree=tree)
    print 'valid set'
    print generate_trivial_tree_dataset_debug(
        os.path.join(allnli_dir, allnli_valid_fname), 
        os.path.join(target_dir, 'allnli_valid_wo-p_{}.json'.format(tree)), 
        trivial_tree=tree)