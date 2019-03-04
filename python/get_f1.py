'''
compute real parsing f1 score given a report file (with pair ID)
'''
import json
import sys
from spinn.util.data import get_brackets
import re
from nltk.tree import Tree

# check spinn F1
rl_exp_name = 'rl_ft_prpn_up_allnli_w-p_20_09' # rl_ft_allnli_wo-p_balanced_00
sl_exp_name = 'sl_rl_prpn_up_allnli_w-p_24_00'
# gumbel_exp_name = 'gumbel_04'
prediction_fname = 'PATH/data/tree_distillation/allnli_wp_dev_gumbel/\
' + rl_exp_name + '.eval_set_0.report'
prpn_fname = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance/\
prpn_up_allnli_w-p_20_valid.json'

def check_right_branching(l):
    if not isinstance(l[0], unicode):
        return False
    elif isinstance(l[1], list):
        return check_right_branching(l[1])
    elif isinstance(l[1], unicode):
        return True
    else:
        return False

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

def tree_list2string(tree):
    '''
    for pretty print
    '''
    if isinstance(tree, list):
        s = '( X '
        for x in tree:
            s += tree_list2string(x)
            s += ' '
        s += ' )'
        return s
    else:
        return ' (X ' + tree + ' ) '

prpn_f1_list = []
pred_f1_list = []
x = 0
with open(prpn_fname, 'r') as fr1:
    with open(prediction_fname, 'r') as fr2:
        prpn_lines = fr1.readlines()
        pred_lines = fr2.readlines()
        print 'number of prpn: ', len(prpn_lines)
        print 'number of pred: ', len(pred_lines)
        i = 0
        j = 0
        s = 0

        while i < len(prpn_lines) and j < len(pred_lines):
            # read prpn example
            prpn_example = json.loads(prpn_lines[i])
            prpn_key = prpn_example['pairID']

            # read prediction example
            pred_example = json.loads(pred_lines[j])
            pred_key = pred_example['example_id']

            if prpn_key == pred_key:
                prpn_f1_list.append(
                    compute_f1_from_list(prpn_example['sentence1_prpn_binary_parse'],
                    prpn_example['sentence1_binary_parse']
                ))
                prpn_f1_list.append(
                    compute_f1_from_list(prpn_example['sentence2_prpn_binary_parse'],
                    prpn_example['sentence2_binary_parse']
                ))
                pred_f1_list.append(
                    compute_f1_from_list(pred_example['sent1_tree'],
                    prpn_example['sentence1_binary_parse']
                ))
                pred_f1_list.append(
                    compute_f1_from_list(pred_example['sent2_tree'],
                    prpn_example['sentence2_binary_parse']
                ))                

                i += 1
                j += 1 
                s += 1
            else:
                i += 1

print '=' * 30
print 'number of matching', s
print 'prpn f1 {:.4f}'.format(sum(prpn_f1_list) / len(prpn_f1_list))
print 'prediction f1 {:.4f}'.format(sum(pred_f1_list) / len(pred_f1_list))
print '*' * 50
