#!/bin/bash

# L1 auto
 bin/./VeST sample/ML_small.tensor NA sample/result/ 4 10 10 5 5 L1


# L1 manual
bin/./VeST sample/ML_small.tensor NA sample/result/ 4 10 10 5 5 L1 -m 0.8

# LF auto
bin/./VeST sample/ML_small.tensor NA sample/result/ 4 10 10 5 5 LF 

# LF manual
bin/./VeST sample/ML_small.tensor NA sample/result/ 4 10 10 5 5 LF -m 0.8




