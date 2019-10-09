This branch includes the backend vectorization for the Arm Scalable vector extension (SVE).

To compile the code make sure you are using the armclang compiler version 19.3.
Earlier versions include fatal bugs.
First specify the arm compiler:

export CC=`which armclang`
export CXX=`which armclang++`

When running cmake include the following options

-DARB_VECTORIZE=ON -DARB_ARCH=armv8-a+sve

SVE vector length must be specified at compile time.
By default Arbor is built for vector length = 256bits.
If one would like to build for other length, one can use:

-DSVE_VECTOR_LENGTH={128, 256, 512, 1024, 2048}

Arbor checks SVE vector length at runtime and exits with EXIT_FAILURE upon mismatch.
