#!/usr/bin/env python
import os
import sys
import cPickle

if __name__ == '__main__':
    assert sys.argv[1] == '--db'
    db = cPickle.load(open(sys.argv[2]))
    for entry in db:
        print entry

