#!/usr/bin/env python
import os
import sys

from build_csv import build_results

if __name__ == '__main__':
    r = build_results(sys.argv[1])
    keys = r.keys()
    keys.sort()

    for k in keys:
        v = r[k]
        print k
        r_k = [(v[i],i) for i in v]
        r_k.sort()
        r_k.reverse()
        for t, i in r_k:
            print "   %10.2f - %s" %(t, i)
        print ''
