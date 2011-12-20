#!/bin/bash

# HACKS TO USE OPENBLAS
export LIBRARY_PATH=./lib:~/.VENV/base/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=./lib:~/.VENV/base/lib:$LD_LIBRARY_PATH

#-convfast use "fast" convolution code instead of standard [false]
#-openmp   use openmp *package* [false]
#-double   use doubles instead of floats [false]
#-cuda     use CUDA instead of floats [false]
#-batch    batch size [1]
#-gi       compute gradInput [false]
#-v        be verbose [false]

# this would use GEMM for convolution, Koray said this was not use
# and it makes a huge unrolled matrix for large problems.
USE_CONVFAST=""

for batchsize in 1 10 100 ; do
    for PREC in 32 64 ; do
        if true ; then
            OUTPUT=run.sh.results_${HOSTNAME}_b${batchsize}_p${PREC}
            echo "Running normal" $OUTPUT
            echo "host=$HOSTNAME" > "$OUTPUT"
            echo "device=CPU" >> "$OUTPUT"
            echo "OpenMP=0" >> "$OUTPUT"
            echo "batch=$batchsize" >> "$OUTPUT"
            echo "precision=$PREC" >> "$OUTPUT"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi

            ~/local/bin/lua benchmark.lua -batch $batchsize $USE_DOUBLE >> "$OUTPUT"
        fi

        if true ; then
            OUTPUT=run.sh.results_${HOSTNAME}_b${batchsize}_p${PREC}_openmp
            echo "Running OpenMP " $OUTPUT
            echo "host=$HOSTNAME" > "$OUTPUT"
            echo "device=CPU" >> "$OUTPUT"
            echo "OpenMP=1" >> "$OUTPUT"
            echo "batch=$batchsize" >> "$OUTPUT"
            echo "precision=$PREC" >> "$OUTPUT"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi
            ~/local/bin/lua benchmark.lua -batch $batchsize $USE_DOUBLE -openmp >> "$OUTPUT"
        fi

        if true ; then
            OUTPUT=run.sh.results_${HOSTNAME}_b${batchsize}_p${PREC}_cuda
            echo "Running CUDA " $OUTPUT
            echo "host=$HOSTNAME" > "$OUTPUT"
            echo "device=GTX480" >> "$OUTPUT"
            echo "OpenMP=0" >> "$OUTPUT"
            echo "batch=$batchsize" >> "$OUTPUT"
            echo "precision=32" >> "$OUTPUT"
            if [ $PREC = 32 ] ; then
                USE_DOUBLE=""
            else
                USE_DOUBLE="-double"
            fi
            ~/local/bin/lua benchmark.lua -batch $batchsize $USE_DOUBLE -cuda >> "$OUTPUT"
        fi
    done
done


