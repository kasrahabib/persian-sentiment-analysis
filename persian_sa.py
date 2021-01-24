from __future__ import division, print_function, unicode_literals
import sklearn
from CommentToWordCounterTransformer import CommentToWordCounterTransformer
from WordCounterToVectorTransformer import WordCounterToVectorTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle, os


np.random.seed(42)

svm_model = os.path.join(os.path.dirname(__file__), './resources/svm_model.pkl')
preprocessing_pipline = os.path.join(os.path.dirname(__file__), './resources/preprocessing_pipline.pkl')

svm_model = pickle.load(open(svm_model, 'rb'))
preprocess_pipeline = pickle.load(open(preprocessing_pipline, 'rb'))

def predict_sentiment(new_comment, return_class_label = False):
    p_class = svm_model.predict(preprocess_pipeline.transform(new_comment).toarray())
    if return_class_label:
        return p_class[0]
    return "Positive!" if p_class[0] > 0 else "Negative!"

print('''

This app uses ML to predict setntiment (e.g., Positive or Negative)
of a given Persian text. Toexit  the app write  'exit' in terminal.

''')
while(True):
    to_predict = input('Input: ')
    if not to_predict:
        print('No input to predict sentiment!')
        print('\n')
        continue;

    elif to_predict:
        if (str(to_predict) == str('exit')):
            print('...  exit: 0')
            break;
        print('...', predict_sentiment(np.array([to_predict])))
        print('\n')
        continue;
    else:
        pass
