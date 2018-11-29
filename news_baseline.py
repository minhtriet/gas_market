from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from util import data_generator

for shift in range(5, 7):
    print(shift)
    x_train, x_test, y_train, y_test = data_generator.baseline_crf(train_percentage=0.6, sft=shift, future=True)
    crf = ChainCRF(n_states=2, n_features=x_train.shape[1])
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)
    ssvm = OneSlackSSVM(model=crf, C=.1, max_iter=10)
    ssvm.fit([x_train], [y_train])
    print(ssvm.score([x_test], [y_test]))
