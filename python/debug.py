'''
compute real parsing f1 score given a report file (without pair ID)
'''
import json
import sys
from spinn.util.data import get_brackets
import re

# # check empty sentences
# target_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance/'
# fname = 'allnli_wo-p_right_branching_train_shuffled_valid.json'
# num = 0
# with open(target_dir + fname, 'r') as fr:
#     for line in fr:
#         loaded_example = json.loads(line)
#         s1 = loaded_example["sentence1"]
#         s2 = loaded_example["sentence2"]
#         if len(s1) == 0 or len(s2) == 0:
#             num += 1
# print 'empty sentences', num

# check spinn F1
rl_exp_name = 'rl_ft_allnli_wo-p_right_branching_05' # rl_ft_allnli_wo-p_balanced_00
sl_exp_name = 'sl_rl_definition_prpn_up_allnli_wo-p_04_01'
prediction_fname = 'PATH/data/tree_distillation/allnli_dev_gumbel/\
' + rl_exp_name + '.eval_set_0.report'
prpn_fname = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance/\
allnli_wo-p_balanced_valid.json'

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
            prpn_key = re.sub('[0-9]+', 'N', ' '.join(prpn_example['sentence1'])) + \
                ' || ' + \
                re.sub('[0-9]+', 'N', ' '.join(prpn_example['sentence2']))
            prpn_key = prpn_key.encode('UTF-8').lower()

            # read prediction example
            pred_example = json.loads(pred_lines[j])
            pred_key = list2words(pred_example['sent1_tree']) + ' || ' + list2words(pred_example['sent2_tree'])
            pred_key = pred_key.lower()

            if check_key(prpn_key, pred_key):
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
                # print '*' * 50
                # print prpn_key
                # print pred_key
                # x += 1
                # if x > 10:
                #     break
                i += 1

print '=' * 50
print 'number of matching', s
print 'prpn f1 {:.4f}'.format(sum(prpn_f1_list) / len(prpn_f1_list))
print 'prediction f1 {:.4f}'.format(sum(pred_f1_list) / len(pred_f1_list))
