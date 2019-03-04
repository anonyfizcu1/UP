'''
get agreement (parsing F1) against right-branching
'''
import json
import sys
from spinn.util.data import get_brackets
import re
import os.path
import os

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
    # elif isinstance(l[0], str) or isinstance(l[0], unicode):
    #     return l[0] + ' ' + list2words(l[1])
    # elif isinstance(l[1], str) or isinstance(l[1], unicode):
    #     return list2words(l[0]) + ' ' + l[1]
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

def get_f1_against_right_branching_from_list(tree_list):
    t_brackets, length = get_brackets(tree_list)
    t_brackets.add((0, length))
    rb_brackets = {(x, length) for x in range(length-1)}
    return compute_f1(t_brackets & rb_brackets, t_brackets, rb_brackets)

def get_f1_against_right_branching_from_fpath(fpath, prpn=False):
    pred_f1_list = []
    with open(fpath, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            # read prediction example
            example = json.loads(line)
            if prpn:
                t_l1 = example['sentence1_prpn_binary_parse']
                t_l2 = example['sentence2_prpn_binary_parse']
            else:
                t_l1 = example['sent1_tree']
                t_l2 = example['sent2_tree']                
            pred_f1_list.append(get_f1_against_right_branching_from_list(t_l1))
            pred_f1_list.append(get_f1_against_right_branching_from_list(t_l2))                
    print '=' * 50
    f1 = sum(pred_f1_list) / len(pred_f1_list)
    print '\t{} {:d}'.format(fpath.split('/')[-1], len(lines))
    print '\tprediction f1 against RB {:.4f}'.format(f1)
    return f1

report_name = 'gumbel'
report_dir = 'PATH/data/tree_distillation/allnli_wp_dev_gumbel/rl_eval_report/'
prpn_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'

prpn_list = []
for i in range(5):
    prpn_list.append(os.path.join(prpn_dir, 'prpn_up_allnli_wo-p_0{}_valid.json'.format(i)))
report_list = []
for fname in os.listdir(report_dir):
    if report_name in fname:
        report_list.append(os.path.join(report_dir, fname))

# f_list = prpn_list
f_list = report_list
print len(f_list)

f1_against_rb = []
for fpath in f_list:
    f1_against_rb.append(get_f1_against_right_branching_from_fpath(fpath, prpn=False))
print '=' * 50
print 'on average {:.3f}'.format(sum(f1_against_rb) / len(f1_against_rb))