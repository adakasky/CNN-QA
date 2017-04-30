"""
preprocess the data of MS-MARCO

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""
import utils

utils.preprocess_squad()
utils.prepare_squad()

# utils.preprocess_squad("../data/train-v1.1.json", "../data/char_squad_train.json", "../data/squad_train.txt")
# utils.prepare_squad("../data/char_squad_train.json", "../data/squad_train.txt", "../data/prepared_squad_train.txt")
