'''
parsing performance on different tags given pair ID w/ punctuations
'''

import json
import sys
from spinn.util.data import get_brackets
import re
import os
import nltk

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']

def list2words(l):
    if isinstance(l, list) and len(l) == 0:
        raise ValueError('empty sentence!')
    elif isinstance(l, list) and len(l) == 1:
        return l[0]
    elif isinstance(l, unicode): 
        return l.encode('UTF-8')
    elif isinstance(l, str): 
        return l
    else:
        return list2words(l[0]) + ' ' + list2words(l[1])

def compute_f1(overlap, std_tree, model_tree):
    prec = float(len(overlap)) / (len(model_tree) + 1e-8)
    reca = float(len(overlap)) / (len(std_tree) + 1e-8)
    if len(std_tree) == 0:
        reca = 1.
        if len(model_tree) == 0:
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1

def compute_f1_from_list(l1, l2):
    t1_brackets, len1 = get_brackets(l1)
    t2_brackets, len2 = get_brackets(l2)
    assert len1 == len2
    t1_brackets.add((0,len1))
    t2_brackets.add((0,len2))
    return compute_f1(t1_brackets & t2_brackets, t1_brackets, t2_brackets) 

def filter_words(tree):
    words = []
    for w, tag in tree.pos():
        if tag in word_tags + punctuation_tags:
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            # if tag == 'CD':
            #     w = 'N'
            words.append(w)
    return words

def tree2list(tree):
    if isinstance(tree, nltk.Tree):
        if tree.label() in word_tags + punctuation_tags:
            return tree.leaves()[0]
        elif tree.label() in punctuation_tags:
            pass
        else:
            root = []
            for child in tree:
                c = tree2list(child)
                if c != []:
                    root.append(c)
            if len(root) > 1:
                root[0] = root[0]
                return root
            elif len(root) == 1:
                return root[0] + tree.label()
    return []

def get_tag_brackets(tree):
    brackets = []
    def get_tag_span(tree, start_idx, brackets):
        end_idx = start_idx
        if isinstance(tree, nltk.Tree):
            label = tree.label()
            if label not in word_tags + punctuation_tags:
                for child in tree:
                    end_idx = get_tag_span(child, end_idx, brackets)
                brackets.append((label, (start_idx, end_idx)))
                if end_idx == start_idx + 1:
                    brackets.pop()
                return end_idx
            elif label in word_tags + punctuation_tags:
                return start_idx + 1
            else:
                return start_idx
        else:
            return 0
    get_tag_span(tree, 0, brackets)
    return brackets

def get_tag_acc_from_fpath(fpath_dev, fpath_pred, tag_dev_freq, tag_pred_freq, prpn=True):
    with open(fpath_pred, 'r') as fr_pred:
        with open(fpath_dev, 'r') as fr_dev:
            lines_pred = fr_pred.readlines()
            lines_dev = fr_dev.readlines()
            i = 0
            j = 0
            s = 0
            x = 0
            while i < len(lines_dev) and j < len(lines_pred):                
                # read dev example
                example_dev = json.loads(lines_dev[i])
                key_dev = example_dev['pairID']

                # read prediction example
                example_pred = json.loads(lines_pred[j])
                if not prpn:
                    key_pred = example_pred['example_id']
                else:
                    key_pred = example_pred['pairID']

                if key_dev == key_pred:
                    tree1_dev = nltk.Tree.fromstring(example_dev['sentence1_parse'])
                    tree2_dev = nltk.Tree.fromstring(example_dev['sentence2_parse'])
                    tag_brackets1_dev = get_tag_brackets(tree1_dev)
                    tag_brackets2_dev = get_tag_brackets(tree2_dev)

                    if not prpn:
                        brackets1_pred, len1 = get_brackets(example_pred['sent1_tree'])
                        brackets2_pred, len2 = get_brackets(example_pred['sent2_tree'])
                    else:
                        brackets1_pred, len1 = get_brackets(example_pred['sentence1_prpn_binary_parse'])
                        brackets2_pred, len2 = get_brackets(example_pred['sentence2_prpn_binary_parse'])                        

                    brackets1_pred.add((0,len1))
                    brackets2_pred.add((0,len2))

                    for tag, bracket in tag_brackets1_dev:
                        if tag not in tag_dev_freq:
                            tag_dev_freq[tag] = 1
                        else:
                            tag_dev_freq[tag] += 1                           
                        if bracket in brackets1_pred:
                            if tag not in tag_pred_freq:
                                tag_pred_freq[tag] = 1
                            else:
                                tag_pred_freq[tag] += 1     

                    for tag, bracket in tag_brackets2_dev:
                        if tag not in tag_dev_freq:
                            tag_dev_freq[tag] = 1
                        else:
                            tag_dev_freq[tag] += 1    
                        if bracket in brackets2_pred:
                            if tag not in tag_pred_freq:
                                tag_pred_freq[tag] = 1
                            else:
                                tag_pred_freq[tag] += 1               
                    i += 1
                    j += 1 
                    s += 1
                else:
                    i += 1   
            print '\t{}: {}/{}|{}'.format(fpath_pred.split('/')[-1], len(lines_dev), len(lines_pred), s)

# check spinn F1
report_name = 'rl_ft'
dev_fpath = 'PATH/data/nli/all_nli/multinli_1.0_dev_matched.jsonl'
report_dir = 'PATH/data/tree_distillation/allnli_wp_dev_gumbel/rl_eval_report/'
report_list = []
for fname in os.listdir(report_dir):
    if report_name in fname:
        report_list.append(os.path.join(report_dir, fname))

prpn_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'
prpn_list = []
for i in range(5):
    prpn_list.append(os.path.join(prpn_dir, 'prpn_up_allnli_w-p_2{}_valid.json'.format(i)))

f_list = report_list
# f_list = prpn_list

tag_dev_freq = {}
tag_pred_freq = {}

for fpath in f_list:
    get_tag_acc_from_fpath(dev_fpath, fpath, tag_dev_freq, tag_pred_freq, prpn=False)

with open('tag_acc.csv', 'w') as fw:
    for tag in tag_pred_freq:
        tag_pred_freq[tag] = tag_pred_freq[tag] / (tag_dev_freq[tag] + 0.0)
        fw.write('{}, {:.3f}, {:.3f}'.format(tag, tag_pred_freq[tag], tag_dev_freq[tag] / (len(f_list) + 0.0) ))
        fw.write('\n')




