This branch includes the backend vectorization for the Arm Scalable vector extension (SVE) using GCC10 fixed size SVE types.

To compile the code make sure you are using GCC 10.
First specify the gcc compiler:

export CC=`which gcc`
export CXX=`which g++`

When running cmake include the following options

-DARB_VECTORIZE=ON -DARB_ARCH=armv8-a+sve

SVE vector length must be specified at compile time.
By default Arbor is built for vector length = 256bits.
If one would like to build for other length, one can use:

-DSVE_VECTOR_LENGTH={128, 256, 512, 1024, 2048}
