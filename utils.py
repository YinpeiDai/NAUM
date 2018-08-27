import re,pprint
from nltk.corpus import stopwords
import nltk
import json
import os
import numpy as np
import copy
import xlwt
from collections import defaultdict



def meaningless_words():
    '''
    bag-of-words feature can be reinforced by deleting stopwords in SVM and LR
    '''
    stopwords_list = []
    for word in stopwords.words('english'):
        tokens = nltk.word_tokenize(word)
        stopwords_list += tokens
    stopwords_list = list(set(stopwords_list)) + stopwords.words('english')
    return stopwords_list


def normalize(text):
    '''
    This is for cleaning sentences
    '''

    # deal with some spell error
    text = re.sub(r'dien\'t', 'did not', text)
    text = re.sub(r' y/o ', ' year old ', text)
    text = re.sub(r' dayy ', ' day ', text)
    text = re.sub(r' sumhow ', ' somehow ', text)
    text = re.sub(r' juss ', ' just ', text)
    text = re.sub(r' wiil ', ' will ', text)
    text = re.sub(r' kry ', ' cry ', text)
    text = re.sub(r' messeges ', ' messages ', text)
    text = re.sub(r' rigjt ', ' right ', text)
    text = re.sub(r' girlfrined ', ' girlfriend ', text)
    text = re.sub(r' mounths ', ' months ', text)
    text = re.sub(r' togheter ', ' together ', text)
    text = re.sub(r' bieng ', ' being ', text)
    text = re.sub(r' evryone ', ' everyone ', text)
    text = re.sub(r' ingnore ', ' ignore ', text)
    text = re.sub(r'ppppppplllllllleeeeeeeeeeeaaaaaaaaaaassssassseeeeeee', ' please ', text)
    text = re.sub(r' veryyyy ', ' very ', text)
    text = re.sub(r' realllly ', 'really', text)
    text = re.sub(r' [wW]hyyyyy', ' why ', text)
    text = re.sub(r' othr ', ' other ', text)
    text = re.sub(r'T\'was', 'i was', text)
    text = re.sub(r' tommarow ', ' tomarrow ', text)
    text = re.sub(r' funnily ', ' funny ', text)


    # lower case  all words and clean strange symbols
    text = text.lower()
    text = text.replace('\n',' . ')
    text = re.sub(r'[^A-z0-9!?.,\':&]', ' ', text)
    text = text.replace('_', ' ')

    # deal with num
    text = re.sub(r'(^|\s)\d*?[\'.:]\d+[A-z]*?(\s|$)', ' <NUM> ', text)
    text = re.sub(r'(^|\s)\d*[$£]\d+(\s|$)', ' <NUM> ', text)
    text = re.sub(r'\d+', ' <NUM> ', text)

    # deal with special mark
    text = text.replace('&', ' and ')
    text = re.sub(r':\'\(', ' , ', text)
    text = re.sub(r'[([)\]]', ' ', text)
    text = re.sub(r':[A-Z]', ' ', text)
    text = re.sub(r':','', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'[/\\]', ' ', text)
    text = re.sub(r', \' \.', ' . ', text)
    text = re.sub(r'&+', ' and ', text)
    text = re.sub(r'(,\s*\.)+', ' . ', text)
    text = re.sub(r'(\.\s*,)+', ' . ', text)

    # add space to marks
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'\.', ' . ', text)
    text = re.sub(r'!', ' ! ', text)
    text = re.sub(r'\?', ' ? ', text)
    text = re.sub(r'\n', ' . ', text)

    # deal with repeating marks
    text = re.sub(r'(!\s+)+', ' ! ', text)
    text = re.sub(r'(\?\s*)+', ' ? ', text)
    text = re.sub(r'(\.\s*)+', ' . ', text)
    text = re.sub(r'(,\s*)+', ' , ', text)


    # join together
    text = nltk.word_tokenize(text)  # split original sent
    text = ' '.join(text)
    text = text.replace('< NUM >', '<NUM>')
    return text


def F1_score(pred_prob, true_prob):
    '''
    return P,R,A,F1,TP,FP,TN,FN
    :param pred_prob: predicted probability list
    :param true_prob: true probability list
    :return: F1 score and other metrics
    '''
    TP, FP, FN, TN = 0, 0, 0, 0
    for i, label in enumerate(true_prob):
        if label == 0 and pred_prob[i] <= 0.5:
            TP += 1
        elif label == 0 and pred_prob[i] > 0.5:
            FN += 1
        elif label == 1 and pred_prob[i] <= 0.5:
            FP += 1
        elif label == 1 and pred_prob[i] > 0.5:
            TN += 1
    total_num = len(true_prob)
    assert TP + TN + FP + FN == len(true_prob)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accu = (TP + TN) / (TP + TN + FP + FN)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    other_metrics = precision, recall, accu, TP / total_num, FP / total_num, TN / total_num, FN / total_num
    return f1_score, other_metrics


def sub_UNK(sent, word_dict):
    words = sent.split()
    for i, w in enumerate(words):
        if w not in word_dict:
            words[i] = '<UNK>'
    return ' '.join(words)

def generate_configuration(config):
    '''
    transform some str values into int or list
    :param config: class configparser
    :return: dict
    '''
    configuration = {}
    for sec_name, section in config.items():
        configuration[sec_name] = {}
        for k, v in section.items():
            if re.match(r'\d+\.\d+', v):
                configuration[sec_name][k] = float(v)
            elif re.match(r'\d+', v):
                configuration[sec_name][k] = int(v)
            elif re.match(r'\[.+\]', v):
                try:
                    configuration[sec_name][k] = \
                        list([int(i) for i in v.replace("[", "").replace("]", "").replace(" ", "").split(",")])
                except:
                    configuration[sec_name][k] = \
                        list([i for i in v.replace("[", "").replace("]", "").replace(" ", "").split(",")])
            else:
                configuration[sec_name][k] = v
    return configuration


def find_super_category(ontology, sub_category):
    for k , vs in ontology.items():
        if sub_category in vs:
            return k


def generate_results(model, vector_size, **kwargs):
    '''
    load all the result files
    output complete results for this model
    there should be num_of_seeds * num_of_labels * num_oversamplingratio * num_of_rounds files
    please using this function when you get all the results.
    '''

    with open('Data/CBT_ontology.json') as f:
        CBT_ontology = json.load(f)
    all_labels = CBT_ontology['emotions'] + \
                 CBT_ontology['situations'] + CBT_ontology['thinking_errors']
    label_count = {'Anger': 595,
                   'Anxiety': 2547,
                   'Bereavement': 107,
                   'Black_and_white': 840,
                   'Blaming': 325,
                   'Catastrophising': 479,
                   'Comparing': 132,
                   'Depression': 836,
                   'Disqualifying_the_positive': 248,
                   'Emotional_reasoning': 537,
                   'Existential': 885,
                   'Fortune_telling': 1037,
                   'Grief': 230,
                   'Guilt': 136,
                   'Health': 428,
                   'Hurt': 802,
                   'Inflexibility': 326,
                   'Jealousy': 126,
                   'Jumping_to_negative_conclusions': 1782,
                   'Labelling': 424,
                   'Loneliness': 299,
                   'Low_frustration_tolerance': 647,
                   'Mental_filtering': 222,
                   'Mind-reading': 589,
                   'Other': 223,
                   'Over-generalising': 512,
                   'Personalising': 236,
                   'Relationships': 2727,
                   'School_College': 334,
                   'Shame': 229,
                   'Work': 246}

    complete_results = {}
    for metric in ['Precision', 'Recall', 'F1_score', 'Accuracy', 'TP', 'FP', 'TN', 'FN']:
        complete_results[metric] = {}
        for label in all_labels:
            complete_results[metric][label] = {'oversampling_ratio1': [],
                                               'oversampling_ratio3': [],
                                               'oversampling_ratio5': [],
                                               'oversampling_ratio7': [],
                                               'oversampling_ratio0': []}

    save_dir = kwargs['save_dir']
    if model in ['LR_BOW', 'SVM_BOW']:
        saved_results_dir = os.path.join(save_dir, '%s_Results' % model)
        output_metrics_filename = 'Complete_results_for_%s.xls' % model

    else:
        saved_results_dir = os.path.join(save_dir, '%s_%dd_Results'%(model, vector_size))
        output_metrics_filename = 'Complete_results_for_%s_%dd.xls' % (model, vector_size)

    for seed in os.listdir(saved_results_dir):
        for label in all_labels:
            for ratio in [0,1,3,5,7]:
                tmp_pre, tmp_rec, tmp_F1, tmp_acc = [], [], [], []
                tmp_TP, tmp_FP, tmp_TN, tmp_FN = [], [], [], []
                for round_id in range(1,1+kwargs['cross_validation']):
                    filepath = os.path.join(
                        saved_results_dir,
                        seed,
                        label,
                        'oversampling_ratio%d' % ratio,
                        'round%d' % round_id, 'results.txt')
                    try:
                        with open(filepath) as f:
                            for line in f:
                                m = re.match('.*? test F1 score: (\d)\.(\d{4})===.*', line)
                                if m:
                                    tmp_F1.append(int(m.group(1)) + int(m.group(2)) / 10000)
                                n = re.match(
                                    '.* other test_metrics: pre=(\d)\.(\d+) recall=(\d)\.(\d+) accu=(\d)\.(\d+) TP=(\d)\.(\d+) FP=(\d)\.(\d+) TN=(\d)\.(\d+) FN=(\d)\.(\d+)===.*',
                                    line)
                                if n:
                                    tmp_pre.append(int(n.group(1)) + int(n.group(2)) / 10000)
                                    tmp_rec.append(int(n.group(3)) + int(n.group(4)) / 10000)
                                    tmp_acc.append(int(n.group(5)) + int(n.group(6)) / 10000)
                                    tmp_TP.append(int(n.group(7)) + int(n.group(8)) / 10000)
                                    tmp_FP.append(int(n.group(9)) + int(n.group(10)) / 10000)
                                    tmp_TN.append(int(n.group(11)) + int(n.group(12)) / 10000)
                                    tmp_FN.append(int(n.group(13)) + int(n.group(14)) / 10000)
                    except:
                        print('The results is not complete !')
                        print('can not find file %s'%filepath)
                        exit(0)
                complete_results['F1_score'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_F1))
                complete_results['Precision'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_pre))
                complete_results['Recall'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_rec))
                complete_results['Accuracy'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_acc))
                complete_results['TP'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_TP))
                complete_results['FP'][label]['oversampling_ratio%d'%ratio].append(np.mean(tmp_FP))
                complete_results['TN'][label]['oversampling_ratio%d' % ratio].append(np.mean(tmp_TN))
                complete_results['FN'][label]['oversampling_ratio%d' % ratio].append(np.mean(tmp_FN))


    wb = xlwt.Workbook()
    for k, item in complete_results.items():
        ws = wb.add_sheet(k)
        write_excel(ws, all_labels, label_count, item, CBT_ontology)

    wb.save(os.path.join(save_dir, output_metrics_filename))


def write_excel(ws, all_labels, label_count, complete_results_metric, CBT_ontology):
    ws.write(0, 0, 'label')
    ws.write(0, 1, 'Freq')
    ws.write(0, 2, 'ratio 1')
    ws.write(0, 3, 'ratio 3')
    ws.write(0, 4, 'ratio 5')
    ws.write(0, 5, 'ratio 7')
    ws.write(0, 6, 'no ratio')

    AVG_F1_mean, AVG_F1_std = defaultdict(list), defaultdict(list)
    weighted_AVG_F1_mean, weighted_AVG_F1_std = defaultdict(list), defaultdict(list)
    Emotion_mean, Emotion_std = defaultdict(list), defaultdict(list)
    Situation_mean, Situation_std = defaultdict(list), defaultdict(list)
    ThinkingError_mean, ThinkingError_std = defaultdict(list), defaultdict(list)

    for i, label in enumerate(all_labels):
        ws.write(i + 1, 0, label)
        ws.write(i + 1, 1, label_count[label])

    for i, label in enumerate(all_labels):
        for ratio, number in complete_results_metric[label].items():
            if ratio == 'oversampling_ratio0':
                ws.write(i + 1, 6, '%0.3f±%0.3f' % (np.mean(number), np.std(number)))
            elif ratio == 'oversampling_ratio1':
                ws.write(i + 1, 2, '%0.3f±%0.3f' % (np.mean(number), np.std(number)))
            elif ratio == 'oversampling_ratio3':
                ws.write(i + 1, 3, '%0.3f±%0.3f' % (np.mean(number), np.std(number)))
            elif ratio == 'oversampling_ratio5':
                ws.write(i + 1, 4, '%0.3f±%0.3f' % (np.mean(number), np.std(number)))
            else:
                ws.write(i + 1, 5, '%0.3f±%0.3f' % (np.mean(number), np.std(number)))
            AVG_F1_mean[ratio].append(np.mean(number))
            AVG_F1_std[ratio].append(np.std(number))
            weighted_AVG_F1_mean[ratio].append(np.mean(number) * label_count[label])
            weighted_AVG_F1_std[ratio].append(np.std(number) * label_count[label])
            if label in CBT_ontology['emotions']:
                Emotion_mean[ratio].append(np.mean(number))
                Emotion_std[ratio].append(np.std(number))

            elif label in CBT_ontology['situations']:
                Situation_mean[ratio].append(np.mean(number))
                Situation_std[ratio].append(np.std(number))
            else:
                ThinkingError_mean[ratio].append(np.mean(number))
                ThinkingError_std[ratio].append(np.std(number))

    ws.write(len(all_labels) + 4, 0, 'AVG F1')
    for ratio, idx in zip([0, 1, 3, 5, 7], [6, 2, 3, 4, 5]):
        ws.write(35, idx, '%0.3f±%0.3f' % (np.mean(AVG_F1_mean['oversampling_ratio%d' % ratio]),
                                           np.mean(AVG_F1_std['oversampling_ratio%d' % ratio])))

    ws.write(len(all_labels) + 5, 0, 'weighted AVG F1')
    for ratio, idx in zip([0, 1, 3, 5, 7], [6, 2, 3, 4, 5]):
        ws.write(36, idx, '%0.3f±%0.3f' % (
        np.sum(weighted_AVG_F1_mean['oversampling_ratio%d' % ratio]) / np.sum(list(label_count.values())),
        np.sum(weighted_AVG_F1_std['oversampling_ratio%d' % ratio]) / np.sum(list(label_count.values()))))

    ws.write(len(all_labels) + 6, 0, 'Emotion')
    for ratio, idx in zip([0, 1, 3, 5, 7], [6, 2, 3, 4, 5]):
        ws.write(37, idx, '%0.3f±%0.3f' % (np.mean(Emotion_mean['oversampling_ratio%d' % ratio]),
                                           np.mean(Emotion_std['oversampling_ratio%d' % ratio])))

    ws.write(len(all_labels) + 7, 0, 'Situation')
    for ratio, idx in zip([0, 1, 3, 5, 7], [6, 2, 3, 4, 5]):
        ws.write(38, idx, '%0.3f±%0.3f' % (np.mean(Situation_mean['oversampling_ratio%d' % ratio]),
                                           np.mean(Situation_std['oversampling_ratio%d' % ratio])))

    ws.write(len(all_labels) + 8, 0, 'ThinkingError')
    for ratio, idx in zip([0, 1, 3, 5, 7], [6, 2, 3, 4, 5]):
        ws.write(39, idx, '%0.3f±%0.3f' % (np.mean(ThinkingError_mean['oversampling_ratio%d' % ratio]),
                                           np.mean(ThinkingError_std['oversampling_ratio%d' % ratio])))


def generate_predictions(model, vector_size, **kwargs):
    '''
    get the results of oversampling ratio 1:1 due to it's the best
    get the prediction of the labelled data
    using majority voting by multiple seeds
    '''
    def F1_SCORE(true_labels, predict_labels):
        TP = len(true_labels & predict_labels)
        if len(predict_labels) == 0:
            precision = 0
        else:
            precision = TP / len(predict_labels)
        if len(true_labels) == 0:
            recall = 0
        else:
            recall = TP / len(true_labels)
        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        return F1

    with open('Data/CBT_ontology.json') as f:
        CBT_ontology = json.load(f)
    all_labels = CBT_ontology['emotions'] + \
                 CBT_ontology['situations'] + CBT_ontology['thinking_errors']


    save_dir = kwargs['save_dir']

    if model in ['LR_BOW', 'SVM_BOW']:
        saved_results_dir = os.path.join(save_dir, '%s_Results' % model)
        output_predictions_filename = 'Predictions_for_%s.json' % model


    else:
        saved_results_dir = os.path.join(save_dir, '%s_%dd_Results' % (model, vector_size))
        output_predictions_filename = 'Predictions_for_%s_%dd.json' % (model, vector_size)

    predictions = {}
    for seed in os.listdir(saved_results_dir):
        for label in all_labels:
            for round_id in range(1,1+kwargs['cross_validation']):
                filepath = os.path.join(
                    saved_results_dir,
                    seed,
                    label,
                    'oversampling_ratio1',
                    'round%d' % round_id, 'results.txt')
                try:
                    with open(filepath) as f:
                        flag = False
                        for line in f:
                            if 'predictions' in line:
                                flag = True
                                continue
                            m = re.match('(\w{24}) ([01]).*', line)
                            if m and flag:
                                if m.group(1) not in predictions:
                                    predictions[m.group(1)] = {}
                                if label not in predictions[m.group(1)]:
                                    predictions[m.group(1)][label] = []
                                predictions[m.group(1)][label].append(int(m.group(2)))
                except:
                    print('The results is not complete !')
                    print('can not find file %s' % filepath)
                    exit(0)

    with open(kwargs['labelled_data_filepath'], 'r') as f:
        labelled_data = json.load(f)

    predicted_labelled_data = {}
    for ID, pred in predictions.items():
        predicted_labelled_data[ID] = {}
        predicted_labelled_data[ID]['prediction'] = {'emotions':[], 'situations':[], 'thinking_errors':[]}
        for l, count in pred.items():
            if np.mean(count) > 0.5:
                if l in CBT_ontology['emotions']:
                    predicted_labelled_data[ID]['prediction']['emotions'].append(l)
                elif l in CBT_ontology['situations']:
                    predicted_labelled_data[ID]['prediction']['situations'].append(l)
                else:
                    predicted_labelled_data[ID]['prediction']['thinking_errors'].append(l)
    for ID in predicted_labelled_data.keys():
        for this_data in labelled_data:
            if ID != this_data['id']:
                continue
            else:
                predicted_labelled_data[ID]['label'] = copy.deepcopy(this_data['label'])
                predicted_labelled_data[ID]['problem'] = this_data['problem']
                predicted_labelled_data[ID]['negative_take'] = this_data['negative_take']
                predicted_labelled_data[ID]['F1 score'] = F1_SCORE(
                    set(predicted_labelled_data[ID]['label']['emotions']+
                        predicted_labelled_data[ID]['label']['situations'] +
                        predicted_labelled_data[ID]['label']['thinking_errors']),
                    set(predicted_labelled_data[ID]['prediction']['emotions'] +
                        predicted_labelled_data[ID]['prediction']['situations'] +
                        predicted_labelled_data[ID]['prediction']['thinking_errors']))

    predicted_labelled_data = list(predicted_labelled_data.values())
    predicted_labelled_data.sort(key=lambda x: x['F1 score'])
    with open(os.path.join(save_dir,output_predictions_filename), 'w') as f:
        json.dump(predicted_labelled_data, f, indent=2)
    print('mean F1 score for labelled data:', np.mean([x['F1 score'] for x in predicted_labelled_data]))

