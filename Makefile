# make file for linux 
CXX=g++

LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER

OPT = -O2 -mcmodel=medium  -fopenmp -w 

all: VeST 

VeST: src/VeST.cpp  
	$(CXX) $(OPT)  -o bin/$@  $< $(LIB_FLAGS)

demo: 
	bin/./VeST sample/sample_s_train.txt sample/sample_s_test.txt sample/result/ 3 10 10 10 L1 
	bin/./VeST sample/sample_s_train.txt sample/sample_s_test.txt sample/result/ 3 10 10 10 L1 -m 0.8 
	bin/./VeST sample/sample_s_train.txt sample/sample_s_test.txt sample/result/ 3 10 10 10 LF 
	bin/./VeST sample/sample_s_train.txt sample/sample_s_test.txt sample/result/ 3 10 10 10 LF -m 0.8 


.PHONY: clean

clean:
	rm -f bin/VeST

