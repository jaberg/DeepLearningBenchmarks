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

    scores = scores[-8:]

    rect = ax.barh(-.15+np.arange(len(scores)),
            [s for (s,i) in scores],
            0.3, # width
            color='b',
            log=False)

    # add some
    ax.set_title('Preliminary Benchmark Results: %s'% task)
    ax.set_yticklabels(['']+[i for (s,i) in scores], minor=True)
    #ax.set_ylabel('Training Speed (examples/sec)')
    #ax.set_xticks(np.arange(len(scores)), minor=False)
    #ax.set_xticklabels([i[:3] for (s,i) in scores])#, minor=True
    #setp(ax.get_xmajorticklabels(), rotation=90, fontsize=10)

    subplots_adjust(left=.29, bottom=.14, right=.97, top=.91)

    savefig('%s.pdf'%task)

