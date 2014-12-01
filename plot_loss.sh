#! /usr/bin/env python

import sys
import matplotlib.pyplot as pyplot

for filename in sys.argv[1:]:
  with open(filename,'rt') as sf:
    y = []
    for line in sf:
      if 'avg_loss = ' in line:
        y.append(float(line.split(' ')[-1]))
    if len(y):
      pyplot.plot(y)
      pyplot.show()
