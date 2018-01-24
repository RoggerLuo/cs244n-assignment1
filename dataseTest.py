#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment

dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# print(dataset.type)
print(dataset.getRandomContext(5))