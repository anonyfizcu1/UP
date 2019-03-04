'''
self agreement on multiple runs
'''

import json
import sys
from spinn.util.data import get_brackets
import re
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
        
def check_key(s1, s2):
    l1 = s1.split()
    l2 = s2.split()
    set1 = set(l1)
    set2 = set(l2)
    if s1 == s2:
        return True
    elif len(l1) == len(l2) and (set2 - set1 == {'_'}):
        return True
    else:
        False

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

def compute_f1_from_lines(lines1, lines2, self_agreement=False):
    i = 0
    j = 0
    s = 0
    f1_list = []
    while i < len(lines1) and j < len(lines2):
        # read prpn example
        example1 = json.loads(lines1[i])
        if not self_agreement:
            key1 = re.sub('[0-9]+', 'N', ' '.join(example1['sentence1'])) + \
                ' || ' + \
                re.sub('[0-9]+', 'N', ' '.join(example1['sentence2']))
            key1 = key1.encode('UTF-8').lower()
        else:
            key1 = list2words(example1['sent1_tree']) + ' || ' + list2words(example1['sent2_tree'])
            key1 = key1.lower()            

        # read prediction example
        example2 = json.loads(lines2[j])
        key2 = list2words(example2['sent1_tree']) + ' || ' + list2words(example2['sent2_tree'])
        key2 = key2.lower()

        if check_key(key1, key2):
            if not self_agreement:
                f1_list.append(
                    compute_f1_from_list(example2['sent1_tree'], example1['sentence1_binary_parse']
                ))
                f1_list.append(
                    compute_f1_from_list(example2['sent2_tree'],
                    example1['sentence2_binary_parse']
                ))   
            else:
                f1_list.append(
                    compute_f1_from_list(example2['sent1_tree'],
                    example1['sent1_tree']
                ))
                f1_list.append(
                    compute_f1_from_list(example2['sent2_tree'],
                    example1['sent2_tree']
                ))                                

            i += 1
            j += 1 
            s += 1
        else:
            # print '*' * 50
            # print prpn_key
            # print pred_key
            # x += 1
            # if x > 10:
            #     break
            i += 1

    print '\tnumber of matching', s
    return sum(f1_list) / len(f1_list)

def compute_f1_from_fpath(fpath1, fpath2, self_agreement):
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
            x = 0
            f1_list = []
            while i < len(lines1) and j < len(lines2):
                # read prpn example
                example1 = json.loads(lines1[i])
                if not self_agreement:
                    key1 = re.sub('[0-9]+', 'N', ' '.join(example1['sentence1'])) + \
                        ' || ' + \
                        re.sub('[0-9]+', 'N', ' '.join(example1['sentence2']))
                    key1 = key1.encode('UTF-8').lower()
                else:
                    key1 = list2words(example1['sent1_tree']) + ' || ' + list2words(example1['sent2_tree'])
                    key1 = key1.lower()            

                # read prediction example
                example2 = json.loads(lines2[j])
                key2 = list2words(example2['sent1_tree']) + ' || ' + list2words(example2['sent2_tree'])
                key2 = key2.lower()

                if check_key(key1, key2):
                    if not self_agreement:
                        f1_list.append(
                            compute_f1_from_list(example2['sent1_tree'], example1['sentence1_binary_parse']
                        ))
                        f1_list.append(
                            compute_f1_from_list(example2['sent2_tree'],
                            example1['sentence2_binary_parse']
                        ))   
                    else:
                        f1_list.append(
                            compute_f1_from_list(example2['sent1_tree'],
                            example1['sent1_tree']
                        ))
                        f1_list.append(
                            compute_f1_from_list(example2['sent2_tree'],
                            example1['sent2_tree']
                        ))                                

                    i += 1
                    j += 1 
                    s += 1
                else:
                    # print '=' * 50
                    # print key1
                    # print '*' * 30
                    # print key2

                    if len(key1) > len(key2):
                        i += 1
                    else:
                        j += 1
                    # i += 1    
    print '\t {} {}'.format(fpath1.split('/')[-1], fpath2.split('/')[-1])
    print '\t {}|{} matching {} f1 {:.3f}'.format(lines1_len, lines2_len, s, sum(f1_list) / len(f1_list))
    return sum(f1_list) / len(f1_list)

def compute_f1_from_fpath_prpn_self_agreement(fpath1, fpath2):
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
                # read prpn example
                example1 = json.loads(lines1[i])
                key1 = re.sub('[0-9]+', 'N', ' '.join(example1['sentence1'])) + \
                    ' || ' + \
                    re.sub('[0-9]+', 'N', ' '.join(example1['sentence2']))
                key1 = key1.encode('UTF-8').lower()
         

                # read prediction example
                example2 = json.loads(lines2[j])
                key2 = re.sub('[0-9]+', 'N', ' '.join(example2['sentence1'])) + \
                    ' || ' + \
                    re.sub('[0-9]+', 'N', ' '.join(example2['sentence2']))
                key2 = key2.encode('UTF-8').lower()

                if check_key(key1, key2):
                    f1_list.append(
                        compute_f1_from_list(example2['sentence1_prpn_binary_parse'], example1['sentence1_prpn_binary_parse']
                    ))
                    f1_list.append(
                        compute_f1_from_list(example2['sentence2_prpn_binary_parse'], example1['sentence2_prpn_binary_parse']
                    ))                               

                    i += 1
                    j += 1 
                    s += 1
                else:
                    i += 1    
    print '\t {}|{} matching {} f1 {:.3f}'.format(lines1_len, lines2_len, s, sum(f1_list) / len(f1_list))
    return sum(f1_list) / len(f1_list)

# check spinn F1
report_name = 'rl_ft'
dev_fpath = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance/allnli_wo-p_balanced_valid.json'
report_dir = 'PATH/data/tree_distillation/allnli_dev_gumbel/rl_eval_report/'
prpn_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'

prpn_list = []
for i in range(5):
    prpn_list.append(os.path.join(prpn_dir, 'prpn_up_allnli_wo-p_0{}_valid.json'.format(i)))
report_list = []
for fname in os.listdir(report_dir):
    if report_name in fname:
        report_list.append(os.path.join(report_dir, fname))

self_agreement = []
for i in range(len(report_list)):
    for j in range(len(report_list)):
        if i != j and i < j:
            self_agreement.append(compute_f1_from_fpath(report_list[i], report_list[j], self_agreement=True))
            # self_agreement.append(compute_f1_from_fpath_prpn_self_agreement(prpn_list[i], prpn_list[j]))
print '=' * 50
print 'on average {:.3f}'.format(sum(self_agreement) / len(self_agreement))
