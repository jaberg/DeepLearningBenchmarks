#!/usr/bin/env python
import os
import sys

results = []

for root, dirs, files in os.walk('.'):
    for bmark in [f for f in files if f.endswith('.bmark')]:
        for line in open(os.path.join(root,bmark)):
            results.append(line[:-1].split('\t'))

for r in results:
    print ' '.join(r)
