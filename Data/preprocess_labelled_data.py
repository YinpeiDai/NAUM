'''
1. transfer labelled  data into suitable format for classifier training
2. get important statistics, such as max length,
'''


import re, copy
import random, json,os
from utils import normalize, sub_UNK

polished_labels = {
    'Anxiety':'Anxiety',
    'Depression':'Depression',
    'Anger (/frustration)':'Anger',
    'Grief/Sadness': 'Grief',
    'Hurt':'Hurt',
    'Shame':'Shame',
    'Loneliness':'Loneliness',
    'Guilt':'Guilt',
    'Jealousy':'Jealousy',
    'Existential':'Existential',
    'Relationships':'Relationships',
    'Work':'Work',
    'Bereavement':'Bereavement',
    'School/College':'School_College',
    'Health':'Health',
    'Other':'Other',
    'Fortune telling':'Fortune_telling',
    'Jumping to negative conclusions':'Jumping_to_negative_conclusions',
    'Black and white (or all or nothing) thinking':'Black_and_white',
    'Over-generalising':'Over-generalising',
    'Blaming':'Blaming',
    'Labelling':'Labelling',
    'Emotional reasoning':'Emotional_reasoning',
    'Mental filtering':'Mental_filtering',
    'Mind-reading':'Mind-reading',
    'Low frustration tolerance':'Low_frustration_tolerance',
    'Disqualifying the positive':'Disqualifying_the_positive',
    'Inflexibility':'Inflexibility',
    'Catastrophising':'Catastrophising',
    'Personalising':'Personalising',
    'Comparing':'Comparing',
}


class labelledDataParsing:
    def __init__(self,
                 vocab_dict,
                 file_list,
                 super_category_list,
                 CBT_ontology_file):
        self.file_list = file_list
        self.labelled_posts = {}
        with open(CBT_ontology_file, 'r') as f:
            self.CBT_ontology = json.load(f)
        self.super_category = super_category_list

        print('loading the lablled data ... ')
        for i, file in enumerate(self.file_list):
            super_category = self.super_category[i]
            with open(file) as f:
                complete_post = ''
                for line in f:
                    if complete_post == '':
                        complete_post = line.strip()
                    else:
                        complete_post = complete_post + ' ' + line.strip()

                    tokens = complete_post.split('||||')
                    if len(tokens) != 4:
                        continue
                    ID, problem, negative_take, sub_category = \
                        tokens[0], sub_UNK(normalize(tokens[1]),vocab_dict), \
                        sub_UNK(normalize(tokens[2]),vocab_dict), tokens[3]
                    if ID not in self.labelled_posts:
                        self.labelled_posts[ID] = {}
                        self.labelled_posts[ID]['label'] = {
                                'emotions':set(), 'thinking_errors':set(), 'situations':set()
                            }
                    self.labelled_posts[ID]['problem'] = problem
                    self.labelled_posts[ID]['negative_take'] = negative_take
                    if sub_category != 'None':
                        self.labelled_posts[ID]['label'][super_category].add(polished_labels[sub_category])
                    complete_post = ''

        print('Total size of labelled post:', len(self.labelled_posts))

        print('print the data information into data_info.txt ...')
        with open('Data/data_info.txt','w') as f:
            for super_category , values in self.CBT_ontology.items():
                f.write('=== Super category: %s,  %d sub categories totally ===\n' %(super_category, len(values)))
                for sub_category in values:
                    count = self.frequency_count(super_category, sub_category)
                    f.write('%s %d %2.2f\n'%(sub_category, count, count/len(self.labelled_posts)*100))
                f.write('\n')

        for ID, post in self.labelled_posts.items():
            post['id'] = ID
            post['label']['emotions'] = list(post['label']['emotions'])
            post['label']['situations'] = list(post['label']['situations'])
            post['label']['thinking_errors'] = list(post['label']['thinking_errors'])


        print('parsing labelled posts into NAUM_labelled_data.json ... ')

        # we generate the labelled data in a form suitable for skipthought
        # we have to cut sentences.
        NAUM_labelled_data = self.generate_training_data_for_SkipThought()


        with open('Data/NAUM_labelled_data.json', 'w' ) as f: # this is readable form
            json.dump(NAUM_labelled_data, f, indent=2)



    def frequency_count(self, super_category, sub_category):
        count = 0
        for ID, post in self.labelled_posts.items():
            if sub_category in post['label'][super_category]:
                count += 1
        return count

    def generate_training_data_for_SkipThought(self):
        max_length_doc = 0  # max sents one doc contains
        problem_max_length, negative_take_max_length = 0, 0
        testing_data = []
        for ID, post in self.labelled_posts.items():
            problem_max_length = max(problem_max_length,
                                     len(post['problem'].split()))
            negative_take_max_length = max(negative_take_max_length,
                                           len(post['negative_take'].split()))
            one_labelled_data = copy.deepcopy(post)
            split_sentences = []
            for sent in re.split(r'[?!.]', post['problem']):
                sent_words = sent.split()
                sent_len = len(sent_words)
                sent = ' '.join(sent_words)
                if sent_len == 1:
                    if not re.match('^\s*(<NUM>|<UNK>|\d|\w|,)\s*$', sent):
                        split_sentences.append(sent)
                elif sent_len <= 50 and sent_len > 0:
                    split_sentences.append(sent)
                elif sent_len > 50:
                    cutsize = random.choice(list(range(20, 40)))
                    for i in range(sent_len // cutsize):
                        split_sentences.append(' '.join(sent_words[i * cutsize:(i + 1) * cutsize]))
            for sent in re.split(r'[?!.]', post['negative_take']):
                sent_words = sent.split()
                sent_len = len(sent_words)
                sent = ' '.join(sent_words)
                if sent_len == 1:
                    if not re.match('^\s*(<NUM>|<UNK>|\d|\w|,)\s*$', sent):
                        split_sentences.append(sent)
                elif sent_len <= 50 and sent_len > 0:
                    split_sentences.append(sent)
                elif sent_len > 50:
                    cutsize = random.choice(list(range(20, 40)))
                    for i in range(sent_len // cutsize):
                        split_sentences.append(' '.join(sent_words[i * cutsize:(i + 1) * cutsize]))
            one_labelled_data['split_sentences'] = copy.deepcopy(split_sentences)
            testing_data.append(one_labelled_data)
            max_length_doc = max(max_length_doc, len(split_sentences))
        print('     problem contains max %d words' % problem_max_length)
        print('     negative take contains max %d words' % negative_take_max_length)
        print('     After spliting for SkipThought, max number of sents in one doc:', max_length_doc)
        return testing_data


def main(**kwargs):
    with open(kwargs['vocabulary_filepath']) as f:
        vocab_dict = json.load(f)
    all_labelled_posts = os.listdir(kwargs['raw_labelled_data_dir'])
    parser = labelledDataParsing(
        vocab_dict,
        [os.path.join(kwargs['raw_labelled_data_dir'], i) for i in all_labelled_posts],
        [i.replace('post_','') for i in all_labelled_posts],
        'Data/CBT_ontology.json'
    )

if __name__ == '__main__':
    import time
    tstart = time.time()
    main()
    print('extracting labelled posts cost %.1f seconds:' %(time.time() - tstart))