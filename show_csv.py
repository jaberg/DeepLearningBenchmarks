#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import *

def rcolor():
    return tuple(np.random.rand(3))

from build_csv import build_results

results = build_results() # dict task -> impl -> time

n_tasks = len(results)

tasks = results.keys()
impls = set()
for k, v in results.items():
    impls.update(v.keys())

print tasks
print impls
fig = plt.figure()
ax = fig.add_subplot(111)
ind = np.arange(n_tasks)  # the x locations for the groups
width = 0.10       # the width of the bars

rects = []
for i, impl in enumerate(impls):
    means = []
    std = []
    for t in tasks:
        std.append(0)
        try:
            means.append(1.0/results[t][impl])
        except KeyError:
            means.append(0)
    rects.append(ax.bar(ind+i*width, means, width, color=rcolor(), log=True)) #, color='r', yerr=menStd)
    print "adding rect for", impl, means

#womenMeans = (25, 32, 34, 20, 25)
#womenStd =   (3, 5, 2, 3, 3)
#rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)

# add some
ax.set_ylabel('Examples / seconds')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width*len(impls)/2.0)
print 'tasks', tasks
ax.set_xticklabels( [t[:12] for t in tasks] )
print 'gca', gca().get_xticklabels()
setp(gca().get_xticklabels(), rotation=30, fontsize=10)

ax.legend( [r[0] for r in rects], impls, 'upper left' )

#def autolabel(rects):
    # attach some text labels
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                ha='center', va='bottom')
#autolabel(rects1)
#autolabel(rects2)

subplots_adjust(left=.09, bottom=.14, right=.97, top=.95)

plt.show()

