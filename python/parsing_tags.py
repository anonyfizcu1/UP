'''
parsing performance on different tags
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
    print '\t {}|{} matching {} f1 {:.3f}'.format(lines1_len, lines2_len, s, sum(f1_list) / len(f1_list))
    return sum(f1_list) / len(f1_list)

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

def tree2list(tree):
    if isinstance(tree, nltk.Tree):
        if tree.label() in word_tags:
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
            if label not in word_tags and label not in punctuation_tags:
                for child in tree:
                    end_idx = get_tag_span(child, end_idx, brackets)
                brackets.append((label, (start_idx, end_idx)))
                if end_idx == start_idx + 1:
                    brackets.pop()
                return end_idx
            elif label in word_tags:
                return start_idx + 1
            else:
                return start_idx
        else:
            return 0
    get_tag_span(tree, 0, brackets)
    return brackets

# check spinn F1
dev_fpath = 'PATH/data/nli/all_nli/multinli_1.0_dev_matched.jsonl'
report_dir = 'PATH/data/tree_distillation/allnli_dev_gumbel/sl_eval_report/'

prpn_dir = 'PATH/data/tree_distillation/prpn_distance/all_nli_distance'
prpn_list = []
for i in range(5):
    prpn_list.append(os.path.join(prpn_dir, 'prpn_up_allnli_wo-p_0{}_valid.json'.format(i)))

tag_dev_freq = {}
bug = ['if you have any questions about this report please contact henry',
        'if you have any questions regarding this report please call me at']

tag_pred_freq = {}
fpath_list = []
for fname in os.listdir(report_dir):
    if 'gumbel' not in fname:
        fpath_list.append(os.path.join(report_dir, fname))


for fpath in prpn_list:
    with open(fpath, 'r') as fr:
        with open(dev_fpath, 'r') as fr_dev:
            lines_pred = fr.readlines()
            lines_dev = fr_dev.readlines()
            i = 0
            j = 0
            s = 0
            x = 0
            while i < len(lines_dev) and j < len(lines_pred):                
                # read dev example
                example_dev = json.loads(lines_dev[i])
                key_dev = ' '.join(filter_words(nltk.Tree.fromstring(example_dev['sentence1_parse']))) \
                    + ' || ' \
                    + ' '.join(filter_words(nltk.Tree.fromstring(example_dev['sentence2_parse'])))
                key_dev = key_dev.lower()

                skip = False
                for bug_term in bug:
                    if bug_term in key_dev:
                        skip = True
                if skip:
                    i += 1
                    j += 1
                    skip = False
                    continue

                # read prediction example
                example_pred = json.loads(lines_pred[j])
                key_pred = list2words(example_pred['sent1_tree']) + ' || ' + list2words(example_pred['sent2_tree'])
                key_pred = key_pred.lower()

                if check_key(key_dev, key_pred):
                    tree1_dev = nltk.Tree.fromstring(example_dev['sentence1_parse'])
                    tree2_dev = nltk.Tree.fromstring(example_dev['sentence2_parse'])
                    tag_brackets1_dev = get_tag_brackets(tree1_dev)
                    tag_brackets2_dev = get_tag_brackets(tree2_dev)

                    brackets1_pred, len1 = get_brackets(example_pred['sent1_tree'])
                    brackets2_pred, len2 = get_brackets(example_pred['sent2_tree'])
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
                    # x += 1
                    # if x > 500:
                    #     sys.exit(0)
            print '\t{}: {}/{}|{}'.format(fpath.split('/')[-1], len(lines_dev), len(lines_pred), s)

for tag in tag_pred_freq:
    tag_pred_freq[tag] = tag_pred_freq[tag] / (tag_dev_freq[tag] + 0.0)
    print tag, tag_pred_freq[tag]




