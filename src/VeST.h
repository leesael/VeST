/*
 * @file        VEST.h
 *
 * VEST: Very Sparse Factorization of Large-Scale Tensors
 *
 * This software is free of charge under research purposes.
 * For commercial purposes, please contact the author.
 *
 * Recent Updates:
 - None
 * Usage: 
 *   - make demo
 */


/////    Header files     /////

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <armadillo>
#include <omp.h>
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS
using namespace std;
using namespace arma;
char* Data_file;

/**********************
 * GLOBAL VARIABLES
 * ********************/
// Input Arguments with default values 
char* InputPath;                            // ARGV1
char* testInputPath;                        // ARGV2
char* ResultPath;                           // ARGV3
int order = 3;								// Tensor order (e.g., 5)
int *Core_size;								// Core tensor dimensionality
int threadsN = 10;						    // Number of threads to be used
char loss_type = '1';                       // 1 (L1) or F (LF) 
double pratio= 0.9;								// Pruning ratio (used only in manual mode)


// Run settings
bool VERBOSE = false; 
double FIXEDINIT = false;                   // ONLY FOR TESTING WITH FIXED INITIALLIZATION         
bool AUTO = true; 
bool TESTING = false; //true;  
bool FIXED_LAMBDA = false;                  // by default use adjusted lambda value
bool MARK = true;                           // by default mark pruned elements so they are not updated
// Hidden hyperparameters  
int MAX_ITER=100;
double LAMBDA = 5;
double INIT_PR = 0.01;                            // pruning rate step size 
double MAX_PR = 0.1; 
double STOP_RE_DIFF = 0.001; 
double DELTA = 0.05;                       // parameter for determining the elbow point in VeST auto mode
double STOP_RE_P = 0.3;                    // if auto mode stops pruning if current RE becomes larger than MIN_RE*(1.0+STOP_RE_P) 

// Related to input tensor
// Input tensor(Sparse) form : (i1, i2, ... , iN, value)
int *dimensionality, max_dim;				// Tensor dimensionality (e.g., 100x100x100)
int Entries_N;								// Total observable entry number of training tensor set
int test_N;									// Total observable entry number of test tensor set
double *Entries, *E2;						// Containing all values of an input tensor(Train, Test)
int *Index, *I2;							// Containing all indices of an input tensor (Vectorized form) (Train, Test)
int *CountX;								// CountX[n][I_n] contains number of entries of a tensor X whose n-th mode's index is I_n in accumulated form
int *WhereX;					 	       	// WhereX[CountX[n][I_n]] contains all entries of a tensor X whose nth mode's index is I_n
double NormX;								// Norm of input tensor

//Related to factorized component
int Core_N = 1;								// Total number of elements of core tensor
int Core_max=0;
double *CoreTensor;							// Containing all values of a core tensor
int *CorePermu;								// Containing all indices of a core tensor (Vectorized form) 
double MaxCore;								// The maximum value of core tensor

int t_FM_N;									// Total number of nonzeros in a Factor Matrices
int *FM_N;									// number of elements of i-th factor matrix
double *FactorM; 							// Factor matrices in vectorized form

//Pruning-related variables
//Please see the paper and supplementary material for details
int Pruned_Core_N = 0;						// Total number of pruned entries in a core tensor
int* Pruned_FM_N;							// Total number of pruned entries in a factor matrices
bool *markCore;								// Mark pruned or not and when pruned for core tensor	
bool *markFM;								// Mark pruned or not and when pruned for factor matrices

double* f_Resp;								// Containing Resp value of factor matrix entries
int* f_Num;
double* c_Resp;								// Containing Resp value of core tensor entries
int* c_Num;

double totalsparsity = 0;					// total sparsity

// auto-mode variables
bool keepPruning = true;                    
bool alreadyRevived = false;
double RE_max;
double RE_min; 
double sumdRE = 0;
double avgREDiff = 0; 
double preREs[] = {0,0};  


int target_Core_N;                          // target number of elements in a core tensor to prune
int* target_FM_N;                           // target number of elements in a factor matrix to prune



//Update-related variables
int fit_check = 0;
double Error=0, RE, RMSE, MAE, TestRE=0;
double pRE = -1;
int *crows, rowcount;
double *tempCore;
int *Mul, *tempPermu;

// output related
double tZR, cZR, fZR; 
double tTime, avgITime;
int iterNum; 
/* ************************************                 
 * Macros and Static inline functions 
 * ************************************ */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define abss(x) ((x) < 0 ? (-(x)) : (x))
// static inline double abss(double x) { return x > 0 ? x : -x; }

static inline double frand(double x, double y) {//return the random value in (x,y) interval
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

static inline bool comp_c(const int i, const int j) {//used to sort c_Num by value of c_Resp
	return c_Resp[i] > c_Resp[j];
}

static inline bool comp_f(const int i, const int j) {//used to sort f_Num by value of f_Resp
	return f_Resp[i] > f_Resp[j];
}
/* ***************************************
 * FUNCTIONS 
 * *************************************** */
int  Read_Input();
int Read_Training();
void Read_Testing();
void Initialize();
void Initialize_read_init();            // ONLY FOR TESTING WITH FIXED INITIALIZATION
void Update_Core_Tensor();
void Update_Factor_Matrices();
void Reconstruction();
void Calculate_Core_Resp();
void Calculate_FM_Resp();
double Pruning(int g_iter);
void CheckOP(int g_iter, double pR);
void RevivePE(double pR);
void Standardize();
double Test();
double Vest();
void Print();
double Gaussian();  
