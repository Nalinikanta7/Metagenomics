"""MLA - machine-learning-Algorithm
Usage:
    mla.py train <dataset-dir> <model-file> [--vocab-size=<vocab-size>]
    mla.py ask <model-file> <question>
    mla.py (-h | --help)
Arguments:
    <dataset-dir>  Directory with dataset.
    <model-file>   Serialized model file.
    <question>     Text to be classified.
Options:
    --vocab-size=<vocab-size>  Vocabulary size. [default: 10000]
    -h --help                  Show this screen.
"""

from docopt import docopt

##################################################################################

import os
from sklearn.metrics import classification_report

from mla import Model_RF, Dataset

def train_model(dataset_dir, model_file, vocab_size):
    print(f'Training model from directory {dataset_dir}')
    print(f'Vocabulary size: {vocab_size}')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dset = Dataset(train_dir, test_dir)
    X, y = dset.get_train_set()

    model = Model_RF(vocab_size=vocab_size)
    model.train(X, y)

    print(f'Storing model to {model_file}')
    model.serialize(model_file)

    X_test, y_test = dset.get_test_set()
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

def ask_model(model_file, question):
    print(f'Asking model {model_file} about "{question}"')

    model = Model_RF.deserialize(model_file)

    y_pred = model.predict_proba([question])
    print(y_pred[0])

######################################################################
def main():
    arguments = docopt(__doc__)

    if arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>'],
                    int(arguments['--vocab-size'])
        )
    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<question>'])

if __name__ == '__main__':
    main()