# LSTM-QA
Fast and Accurate LSTM-based Comprehension System

Please use Python 3.

The following dependencies are required to run the program:
numpy
tensorflow
keras
h5py
gensim
nltk

Run the code:
1. To preprocess the dataset:
                python preprocess_squad.py
2. Then, train the model use the following script:
               python  lstm_train.py
3. When finish training, use the following script to predict test set:
                python lstm_predict.py
4.  Lastly, use the following script to see the evaluation result:
                python score.py