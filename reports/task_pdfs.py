#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import *

def rcolor():
    return 'b'
    return tuple(np.random.rand(3))

from build_csv import build_results

results = build_results(sys.argv[1]) # dict task -> impl -> time


for task in results:
    print task
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.30       # the width of the bars

    scores = [(s,i) for (i,s) in results[task].items()]
    scores.sort()

    rect = ax.bar(np.arange(len(scores)),
            [s for (s,i) in scores],
            0.3, # width
            color=rcolor(),
            log=True)

    # add some
    ax.set_title('Preliminary benchmark results: %s'% task)
    ax.set_ylabel('examples / second')
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels( [i for (s,i) in scores] )
    setp(gca().get_xticklabels(), rotation=30, fontsize=10)

    subplots_adjust(left=.09, bottom=.14, right=.97, top=.95)

    savefig('%s.pdf'%task)

