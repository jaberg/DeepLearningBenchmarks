"""
Write the results of run.sh to the pickled database of timing results
"""
import os
import sys
import cPickle

def main():
    assert sys.argv[1] == '--db'
    try:
        db = cPickle.load(open(sys.argv[2]))
    except IOError:
        db = []

    for results_file in sys.argv[3:]:
        template = dict()
        for lineno, line in enumerate(open(results_file)):
            if '=' in line:
                key = line[:line.index('=')]
                val = line[line.index('=') + 1:]
                if key in ('host', 'device'):
                    template[key] = val.strip()
                elif key in ('OpenMP',):
                    template[key] = bool(int(val))
                elif key in ('batch', 'precision'):
                    template[key] = int(val)
                else:
                    raise ValueError(key)

            elif line.startswith('mlp'):
                problem, speed_str = line.split('\t')
                entry = dict(template)
                entry['problem'] = problem
                entry['speed'] = float(speed_str)
                db.append(entry)
            elif line.startswith('cnn'):
                problem, speed_str = line.split('\t')
                entry = dict(template)
                entry['problem'] = problem
                entry['speed'] = float(speed_str)
                db.append(entry)
            else:
                print "ERROR: ", line

    if 1:
        print "Writing database to", sys.argv[2]
        cPickle.dump(db, open(sys.argv[2], 'wb'))
    else:
        print "DEBUG FINAL DB:"
        for entry in db:
            print entry

if __name__ == '__main__':
    sys.exit(main())
