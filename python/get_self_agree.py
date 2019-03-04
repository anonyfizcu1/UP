'''
self agreement on multiple runs
'''

import json
import sys
from spinn.util.data import get_brackets
import re
import os


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

def compute_self_agreement_f1_from_fpath(fpath1, fpath2, prpn=False):
    lines1_len = 0
    lines2_len = 0
    s = 0
    with open(fpath1, 'r') as fr1:
        with open(fpath2, 'r') as fr2:
            lines1 = fr1.readlines()
            lines2 = fr2.readlines()
            lines1_len = len(lines1)
            lines2_len = len(lines2)
            i = 0
            j = 0
            f1_list = []
            while i < len(lines1) and j < len(lines2):
                # read two examples
                example1 = json.loads(lines1[i])                
                example2 = json.loads(lines2[j])

                if not prpn:
                    key1 = example1['example_id']
                    key2 = example2['example_id']
                else:
                    key1 = example1['pairID']
                    key2 = example2['pairID']   

                if key1 == key2:
                    if not prpn:
                        f1_list.append(
                            compute_f1_from_list(example2['sent1_tree'],
                            example1['sent1_tree']
                        ))
                        f1_list.append(
                            compute_f1_from_list(example2['sent2_tree'],
                            example1['sent2_tree']
                        ))                                
                    else:
                        f1_list.append(
                            compute_f1_from_list(example2['sentence1_prpn_binary_parse'],
                            example1['sentence1_prpn_binary_parse']
                        ))
                        f1_list.append(
                            compute_f1_from_list(example2['sentence2_prpn_binary_parse'],
                            example1['sentence2_prpn_binary_parse']
                        ))                         
                    i += 1
                    j += 1 
                    s += 1
                else:
                    i += 1    
    print '\t {}|{} matching {} f1 {:.3f}'.format(lines1_len, lines2_len, s, sum(f1_list) / len(f1_list))
    return sum(f1_list) / len(f1_list)


# check spinn F1
report_name = 'rl'
report_dir = 'PATH/data/tree_distillation/allnli_wp_dev_gumbel/rl_eval_report/'
prpn_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'

prpn_list = []
for i in range(5):
    prpn_list.append(os.path.join(prpn_dir, 'prpn_up_allnli_w-p_2{}_valid.json'.format(i)))
report_list = []
for fname in os.listdir(report_dir):
    if report_name in fname:
        report_list.append(os.path.join(report_dir, fname))

# f_list = prpn_list
f_list = report_list
print len(f_list)

self_agreement = []
for i in range(len(f_list)):
    for j in range(len(f_list)):
        if i != j and i < j:
            self_agreement.append(compute_self_agreement_f1_from_fpath(f_list[i], f_list[j], prpn=False))
print '=' * 50
print 'on average {:.3f}'.format(sum(self_agreement) / len(self_agreement))
