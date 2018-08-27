'''
All deep learning models contain two parts
1. Train the embedding
2. Train the classifier
non deep learning models only contain classifier
'''
import configparser
import argparse
import os
import json
import warnings
# ignore all kinds of warnings for printing simplicity
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from CNN_GloVe.Word2Vec.run_model import Run_GloVe
from CNN_GloVe.CNN.run_model import Run_CNN_GloVe
from GRU_SkipThought.Sent2Vec.run_model import Run_SkipThought
from GRU_SkipThought.GRU.run_model import Run_GRU_SkipThought
from FNN_PVDM.Doc2Vec.run_model import Run_PVDM
from FNN_PVDM.FNN.run_model import Run_FNN_PVDM
from Non_DL_models.LR.run_model import Run_LR_BOW
from Non_DL_models.SVM.run_model import Run_SVM_BOW
from Data import preprocess_labelled_data, preprocess_unlabelled_data
from utils import generate_predictions
from utils import generate_results
from utils import generate_configuration
from utils import find_super_category


main_dir = os.path.abspath('.')
CBT_ontology_filepath = os.path.join(
    main_dir,
    'Data',
    'CBT_ontology.json'
)

# load ontology file
with open(CBT_ontology_filepath, 'r') as f:
    CBT_ontology = json.load(f)

# load config.ini file
config = configparser.ConfigParser()
config.read(os.path.join(main_dir, 'config.ini'))
config = generate_configuration(config)



# parse the cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument("--parsing", type=str,
                    choices=['unlabelled_data', 'labelled_data'])

parser.add_argument("--model", type=str,
                    choices=[
                        'GloVe', 'SkipThought', 'PVDM',
                        'CNN_GloVe', 'GRU_SkipThought', 'FNN_PVDM',
                        'LR_BOW', 'SVM_BOW'
                    ])

parser.add_argument("--mode", type=str,
                    choices=['training', 'testing', 'generating', 'evaluating'])

parser.add_argument("--vector_size", type=int, default=100)

parser.add_argument("--seed", type=int)

parser.add_argument("--label", type=str,
                    choices=[v for super_category, sub_categories in CBT_ontology.items() for v in sub_categories])

parser.add_argument("--round_id", type=int,
                    choices=[i for i in range(1, 1+config['DEFAULT']['cross_validation'])])

parser.add_argument("--oversampling_ratio", type=int,
                    choices=[0,1,3,5,7])


args = parser.parse_args()
parsing = args.parsing
model = args.model
mode = args.mode
vector_size = args.vector_size
seed = args.seed
label = args.label
round_id = args.round_id
oversampling_ratio = args.oversampling_ratio

# Parsing the raw data
if parsing == 'unlabelled_data':
    preprocess_unlabelled_data.main(**config['DEFAULT'])
    print('done ...')
    exit(0)
elif parsing == 'labelled_data':
    preprocess_labelled_data.main(**config['DEFAULT'])
    print('done ...')
    exit(0)
else:
    if {'NAUM_labelled_data.json','training_data_for_GloVe.txt',
        'training_data_for_SkipThought.json','training_data_for_PVDM.json'}\
            -set(os.listdir(os.path.join(main_dir, 'Data'))) == set():
        pass
    else:
        print('\nthese files are not found:\n'
              '\'NAUM_labelled_data.json\',\'training_data_for_GloVe.txt\','
              '\'training_data_for_SkipThought.json\',\'training_data_for_PVDM.json\''
              '\nplease parsing the raw unlabelled and labelled data first.\n'
              'try these commands:\n'
              'python run.py --parsing=unlabelled_data\n'
              'python run.py --parsing=labelled_data\n')
        exit(0)

# Training embedding models
if model in ['GloVe', 'SkipThought', 'PVDM']:
    if model == 'GloVe':
        run_model = Run_GloVe(vector_size, **config[model])
    elif model == 'SkipThought':
        run_model = Run_SkipThought(vector_size, **config[model])
    else:
        run_model = Run_PVDM(vector_size, **config[model])

    if mode == 'training':
        input_str = input('do you want to train %s embeddings with vector size %d? (y/n)'%(model, vector_size))
        if input_str == 'y' or input_str == 'Y' \
            or input_str == 'yes' or input_str == 'Yes':
            run_model.train()
        else:
            exit(0)
    elif mode == 'testing':
        input_str = input('do you want to test %s embeddings with vector size %d? (y/n)'%(model, vector_size))
        if input_str == 'y' or input_str == 'Y' \
            or input_str == 'yes' or input_str == 'Yes':
            run_model.testing()
        else:
            exit(0)
    elif mode == 'generating':
        run_model.generating()
    else:
        print('\nplease input correct modes for embedding models !\n')

# Training classifier models
elif model in ['CNN_GloVe', 'GRU_SkipThought', 'FNN_PVDM', 'LR_BOW', 'SVM_BOW']:
    if mode == 'evaluating':
        print('generate results into Complete_results_for_%s_%dd.xls ...' % (model, vector_size))
        generate_results(model, vector_size, **config['DEFAULT'])
        print('generate predictions for labelled data into Predictions_for_%s_%dd.json ...' % (model, vector_size))
        generate_predictions(model, vector_size, **config['DEFAULT'])
    elif mode == 'training':
        if model == 'CNN_GloVe':
            run_model = Run_CNN_GloVe(vector_size, **config[model])
        elif model == 'GRU_SkipThought':
            run_model = Run_GRU_SkipThought(vector_size, **config[model])
        elif model == 'FNN_PVDM':
            run_model = Run_FNN_PVDM(vector_size, **config[model])
        elif model == 'LR_BOW':
            run_model = Run_LR_BOW(**config[model])
        else:
            run_model = Run_SVM_BOW(**config[model])
        if label and round_id>0 and oversampling_ratio>=0:
            run_model.train(seed, find_super_category(CBT_ontology, label),
                            label, round_id, oversampling_ratio)
        else:
            print('\nplease input all the arguments ! for example: \n'
                  'python run.py --model=CNN_GloVe  --mode=training --seed=1 '
                  '--label=Anxiety --round_id=1 --oversampling_ratio=1\n')
    else:
        print('\nplease input correct modes (training/evaluating) for classifier models !\n')
else:
    print('\nplease specify a model !\n')






