import argparse
parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('znum', type=int)
parser.add_argument('a')
args = parser.parse_args()
print args
print args