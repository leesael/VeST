# VeST: Very Sparse Tucker Factorization of Large-Scale Tensors 

0. Information 

    Authors: Park MoonJeong (moonjeong94@dgist.ac.kr), Daegu Gyeonbuk Institute of Science and Technology
            Jang Jun-Gi (elnino4@snu.ac.kr), Seoul National University
            Lee Sael (saellee@snu.ac.kr), Seoul National University
    Version : 1
    Date: 2019-03-10
    Main contact regarding the software: MoonKeong Park (moonjeong94@dgist.ac.kr or moonjeong7523@gmail.com)

    This software is free of charge under research purposes. For commercial purposes, please contact Lee Sael.

1. Introduction

VeST is a tensor factorization method for partially observable data to output a very sparse core tensor and factor matrices.
VeST performs initial decomposition, determines unimportant entries in the decomposition results, removes the unimportant entries, and carefully updates the remaining entries.
To determine unimportant entries, we define and use entry-wise `responsibility' for the decomposed results.
The entries are updated iteratively in a coordinate descent manner in parallel for scalable computation.
    
2. Usage
    
    [Step 1]  Install the OpenMP, Armadillo, LAPACK, and BLAS libraries.

        VeST requires Armadillo and OpenMP libraries.

        Armadillo library is available at the lib directory or http://arma.sourceforge.net/download.html.

        Notice that Armadillo needs LAPACK and BLAS libraries, and they are also available at the lib directory.

        OpenMP version 2.0 or higher is required for VeST. (It is installed by default if you use gcc/g++ compiler)

        [Mac] Mac has it's own 'Accelerate' library that runs with Armadillo.
        
        [Mac] You should have headers. Check to see if you have /usr/include directory. If not, install header packages. Mine was found at '/Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg' 

    [Step 2] Compile and run VeST

        If you successfully install all libraries, "make" command will create executable files and execute a demo.

        Minimum arguments that must be provided are 1. path to training tensor 2. path to testing tensor (if none put 'NA') 3. output directory 4. order and dimensions of each order 5. loss type ('L1' or 'LF') in that order. 

        ex) bin/./VeST sample/sample_train.txt sample/sample_test.txt sample/result/ 3 10 10 10 L1

        The default mode is auto-mode with 10 threads. Manual mode can be set by specifying -m with sparsity (0.0-0.99). 
        
        ex) bin/./VeST sample/sample_train.txt sample/sample_test.txt sample/result/ 3 10 10 10 L1 -m 0.8

        Other parameters can be changed by not recommended. Please view the 'main' function of VeST.cpp for more options. 
        
        If you put the command correctly, VEST will write all values of factor matrices and a core tensor in the result directory set by an argument. 
        
        (PLEASE MAKE SURE THAT YOU HAVE A WRITE PERMISSION TO THE RESULT DIRECTORY!)

        ex) result/FACTOR1, result/CORETENSOR

        ** We note that input tensors must follow base-1 indexing and while outputs are based on base-0 indexing.
