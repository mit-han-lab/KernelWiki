#!/usr/bin/bash

QBLKSIZES=( 64 128 )
KBLKSIZES=( 64 128 )
HEADSIZES=( 64 128 256 )
PRECS=( 1 2 )
VERIFY="verify"
VERIFYALL="verify-all"
VERIFYFLAGS=
REFCHECK=false
INPUT1=$1
echo $1
if [[ $INPUT1 == $VERIFY ]];
then
REFCHECK=true
fi

if [[ $INPUT1 == $VERIFYALL ]];
then
VERIFYFLAGS="-DCOPYOUTMM0 -DCOPYOUTMI"
REFCHECK=true
fi

if [[ $2 == "CTA256" ]];
then
QBLKSIZES=( 128 )
fi

for QBLK in "${QBLKSIZES[@]}"
do
for KBLK in "${KBLKSIZES[@]}"
do

CCMD="/usr/local/cuda-12.3/bin/nvcc -D$2 -D$3 -D$4 -D$5 -D$6 $VERIFYFLAGS --use_fast_math -forward-unknown-to-host-compiler -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK  -I../../lib/ -I ../../include -I/home/colfax/jshah/cutlass-3.4/include -I/home/colfax/jshah/cutlass-3.4/examples/common -I"/usr/local/cuda-12.3/include" -I/include -I/examples -I/home/colfax/jshah/cutlass-3.4/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.3/lib64' -Wl,-rpath,'/usr/local/cuda-12.3/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_pipeline"

echo "====================================================================================================================="
echo $CCMD
echo "====================================================================================================================="

`$CCMD`

for PREC in "${PRECS[@]}"
do
for HEAD in "${HEADSIZES[@]}"
do
# VERIFICATION RUN
if [[ $INPUT1 == $VERIFY || $INPUT1 == $VERIFYALL ]];
then
./fmha_forward_pipeline --batch-size=4 --seq-length=4096 --dim-size=2048 --iterations=1 --head-size=$HEAD  --reference-check=true --prec-type=$PREC
fi

#FLOP RUN
if [[ $INPUT1 != $VERIFYALL ]];
then
echo "FLOP RUN BEGIN"
echo "PREC=$PREC, QBLK=$QBLK, KBLK=$KBLK, HEAD=$HEAD, CTA=$2, $3, $4, $5, $6"
./fmha_forward_pipeline --batch-size=4 --seq-length=4096 --dim-size=2048 --iterations=1000 --head-size=$HEAD  --reference-check=false --prec-type=$PREC
echo "FLOP RUN END"
fi
done
done
done
done
