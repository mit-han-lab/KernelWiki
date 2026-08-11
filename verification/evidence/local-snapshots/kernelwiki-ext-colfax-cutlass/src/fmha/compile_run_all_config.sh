#!/usr/bin/bash

QBLKSIZES=( 64 128 )
KBLKSIZES=( 64 128 )
HEADSIZES=( 64 128 256 )
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
CTAFLAG="-DCTA256"
QBLKSIZES=( 128 )
KBLKSIZES=( 64 128 256 )
fi

for QBLK in "${QBLKSIZES[@]}"
do
for KBLK in "${KBLKSIZES[@]}"
do

echo "/usr/local/cuda-12.2/bin/nvcc $VERIFYFLAGS $CTAFLAG --use_fast_math -forward-unknown-to-host-compiler -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK  -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_Q${QBLK}xK${KBLK}${2}"

/usr/local/cuda-12.2/bin/nvcc $VERIFYFLAGS $CTAFLAG --use_fast_math -forward-unknown-to-host-compiler -DQBLKSIZE=$QBLK -DKBLKSIZE=$KBLK -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_Q${QBLK}xK${KBLK}${2}

for HEAD in "${HEADSIZES[@]}"
do
./fmha_forward_Q${QBLK}xK${KBLK}${2} --batch-size=4 --seq-length=4096 --dim-size=2048 --iterations=1000 --head-size=$HEAD  --reference-check=$REFCHECK

done
done
done

