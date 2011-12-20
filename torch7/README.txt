The code in this directory was forked from

https://github.com/andresy/benchmark/commit/cd81345962bc05fe4819a56a675681605ea1587f

Installing Torch 7
------------------

Torch 7 (https://github.com/andresy/torch) is required to run the scripts in
this folder.  I had personal help from Koray to install torch7. It was
straightforward once he convinced me not to use luarocks. Torch7 failed to
find the openblas I installed, so I had to trick it post-compilation by
setting the LD_LIBRARY_PATH to include a symlink with the right name to my
libopenblas.so.  Use ldd on the libTH.so built by torch7 to see what name you
must give to this fake library.


Running timing experiments
--------------------------

The file run.sh produces a number of timing files whose names are of the form
    run.sh.results_${HOSTNAME}_b[1,10,100]_p[32,64][,_openmp,_cuda]

The cuda trials are run on the GPU device 0, and simply fail if no GPU is
present.


Adding results to DB
--------------------

To add the current timing results to ../db.pkl, type:

$ python add_to_db.py --db ../db.pkl  run.sh.results_*
