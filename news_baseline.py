import argparse
import yaml
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from util import data_generator


parser = argparse.ArgumentParser(description='data related parameters')
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--predict_length', type=int, default=3)
args = parser.parse_args()

with open('config.yaml') as stream:
    try:
        config = yaml.load(stream)
        window = config['window']
    except yaml.YAMLError as exc:
        print(exc)

x_train, x_test, y_train, y_test = data_generator.generate(window, train_percentage=0.6, stride=args.stride,
                                                           predict_length=args.predict_length, embed=False)

crf = ChainCRF()
ssvm = FrankWolfeSSVM(model=crf, C=.1, max_iter=10)
ssvm.fit(x_train, y_train)
ssvm.score(x_test, y_test)
