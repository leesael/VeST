/*
 * @file        VEST.cpp
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

#include "VeST.h"


// [Output] abs value of Gaussing random number 
// [Function] Gaussing number generator centered a 0 with 1 std using Marsaglia's polar method 
double Gaussian() {
	static bool hasSpare = false;
	static double spare;
	if(hasSpare) {
		hasSpare = false;
		return 0.0 + 1.0 * spare;
	}
	hasSpare = true;
	static double u, v, s;
	do {
		u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
		v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
		s = u * u + v * v;
	}
	while( (s >= 1.0) || (s == 0.0) );
    s = sqrt(-2.0 * log(s) / s);
	spare = v * s;
    return abss(0.0 + 1.0 * u * s);
} 


//[Input] Given tensor X
//[Output] Updated WhereX and CountX
//[Function] Assign all non-zeros to the corresponding rows, represented as WhereX and CountX
void assign_index() {
	int *tempX = (int *)malloc(sizeof(int)*max_dim*order);
	int pos = 0, i, j, k, l;
    for (int i = 0; i < order; i++) {
		for (int j = 0; j < dimensionality[i]; j++) {
			CountX[i*max_dim + j] = tempX[i*max_dim + j] = 0;
		}
	}
	for (int i = 0; i < Entries_N; i++) {
		for (int j = 0; j < order; j++) {
			k = Index[pos++];
			CountX[j*max_dim + k]++;
			tempX[j*max_dim + k]++;
		}
	}
	pos = 0;
	int now = 0;
	for (int i = 0; i < order; i++) {
		pos = i*max_dim;
		for (int j = 0; j < dimensionality[i]; j++) {
			k = CountX[pos];
			CountX[pos] = now;
			tempX[pos++] = now;
			now += k;
		}
		CountX[pos] = now;
		tempX[pos] = now;
	}
	pos = 0;
	for (int i = 0; i < Entries_N; i++) {
		for (int j = 0; j < order; j++) {
			k = Index[pos++];
			int now = tempX[j*max_dim + k];
			WhereX[now] = i;
			tempX[j*max_dim + k]++;
		}
	}
	free(tempX);
}

//[Input] Metadata + input tensor as a sparse tensor format 
//[Output] Initialized core tensor G and factor matrices A^{(n)} (n=1...N)
//[Function] Getting all information about an input tensor X / Initialize all factor matrices and core tensor.
int Read_Input(){
 	double start_time = clock(); 
	if(VERBOSE) printf("Reading Input and Initializing ...\n");
    
    int rt = Read_Training();
    
    if(rt == -1) return -1;
     
    if(TESTING){ Read_Testing(); }

    if(FIXEDINIT)           // ONLY FOR TESTING WITH FIXED INITIALIZATION 
        Initialize_read_init();
    else
        Initialize(); 	
     
    if(VERBOSE) printf("Elapsed Time for I/O and Initializations:\t%lf\n\n", (clock() - start_time) / CLOCKS_PER_SEC);
}

int Read_Training(){
	// input files for training tensor and testing tensor 
    FILE *ftrain=NULL; 
    char tmp[1005]; // tmp array for reading file
    // initialize dimension and set core sizes 
	dimensionality = (int *)malloc(sizeof(int)*order);
	for (int i = 0; i < order; i++) {
		dimensionality[i] = 0;
		Core_N *= Core_size[i];
	}
	
    // count Entries number of train set tensor 
    // NOTE: need a better way to do this 
	Entries_N=0;
    ftrain = fopen(InputPath, "r");
    while (fgets(tmp, 1005, ftrain)) {
		Entries_N++;
	}
    fclose(ftrain); 
    
	Index = (int *)malloc(sizeof(int)*Entries_N*order);
	Entries = (double *)malloc(sizeof(double)*Entries_N);

	//assign value of train & test set 
	ftrain = fopen(InputPath,"r");

    if(ftrain == NULL){ perror("cannot open input file"); return -1; }
    
    int pos = 0;
	for (int i = 0; i < Entries_N; i++) {
		fgets(tmp, 1005, ftrain);
		int len = strlen(tmp);
		int k = 0, idx = 0, flag = 0;
		double mul = 0.1, val = 0;
		for (int j = 0; j < len; j++) {
			if (tmp[j] == ' ' || tmp[j] == '\t') {
				Index[pos++] = idx - 1;
				
				if (dimensionality[k] < idx) dimensionality[k] = idx;
				idx = 0;
				k++;
			}
			else if (tmp[j] >= '0' && tmp[j] <= '9') {
				if (flag == 1) {
					val += mul*(tmp[j] - '0');
					mul /= 10;
				}
				else idx = idx * 10 + tmp[j] - '0';
			}
			else if (tmp[j] == '.') {
				val += idx;
				flag = 1;
			}
		}
		if(flag==0) val = idx;
		Entries[i] = val;
		NormX += Entries[i] * Entries[i];
	}
    fclose(ftrain);	
    
    // Setting input training tensor parameters
    t_FM_N=0;
	FM_N = (int*)malloc(sizeof(int)*order);
    for (int i = 0; i < order; i++) {
		max_dim = MAX(max_dim, dimensionality[i]); 
        FM_N[i] = dimensionality[i]*Core_size[i];	
		t_FM_N += dimensionality[i]*Core_size[i];	
	}
	max_dim++;		// just give margin (don't care)
	WhereX = (int *)malloc(sizeof(int)*order*Entries_N);
	CountX = (int *)malloc(sizeof(int)*max_dim*order);
	NormX = sqrt(NormX);

	if(VERBOSE) {
        printf("Reading Training Tensor Done.\n\n[METADATA]\nTensor Order: %d\tSize: ", order);
	    for (int i = 0; i < order; i++) {
		    if (i != order - 1) printf("%dx", dimensionality[i]);
		    else printf("%d\t", dimensionality[i]);
	    }
	    printf("Rank: ");
	    for (int i = 0; i < order; i++) {
		    if (i != order - 1) printf("%dx", Core_size[i]);
		    else printf("%d\t", Core_size[i]);
	    }
	    printf("NNZ : %d\tThreads : %d\tNorm : %lf\n", Entries_N, threadsN, NormX);
    } // end VERBOSE
    return 1; 
}

void Read_Testing(){
    // input files for training tensor and testing tensor 
    FILE *ftest=NULL; 
    char tmp[1005]; // tmp array for reading file

	test_N = 0;
	// count Entries number of train & test set tensor 
    // NOTE: need a better way to do this 
    ftest = fopen(testInputPath, "r");
    
    if(ftest == NULL){ TESTING = false; return; }

    while (fgets(tmp, 1005, ftest)) {
		test_N++;
	}
    fclose(ftest); 

	I2 = (int *)malloc(sizeof(int)*test_N*order);
	E2 = (double *)malloc(sizeof(double)*test_N);

	//assign value of test set 
    ftest = fopen(testInputPath,"r");
    int pos = 0;
    for (int i = 0; i < test_N; i++) {
		fgets(tmp, 1005, ftest);
		int len = strlen(tmp);
		int k = 0, idx = 0, flag = 0;
		double mul = 0.1, val = 0;
		for (int j = 0; j < len; j++) {
			if (tmp[j] == ' ' || tmp[j] == '\t') {
				I2[pos++] = idx - 1;
				if (dimensionality[k] < idx) dimensionality[k] = idx;
				idx = 0;
				k++;
			}
			else if (tmp[j] >= '0' && tmp[j] <= '9') {
				if (flag == 1) {
					val += mul*(tmp[j] - '0');
					mul /= 10;
				}
				else idx = idx * 10 + tmp[j] - '0';
			}
			else if (tmp[j] == '.') {
				val += idx;
				flag = 1;
			}
		}
		if(flag==0) val = idx;
		E2[i] = val;
	}
    fclose(ftest);
    return; 
}


void Initialize(){
    if(VERBOSE) printf("Initialize...\n"); 
	// Initialize output factor matrix intermediate data structures
    assign_index();
    FactorM = (double *)malloc(sizeof(double)*order*max_dim*Core_max);
    f_Num = (int *)malloc(sizeof(int)*order*max_dim*Core_max);
	if(!AUTO) target_FM_N = (int *)malloc(sizeof(int)*order);
    Pruned_FM_N = (int *)malloc(sizeof(int)*order);
	markFM = (bool *)malloc(sizeof(int)*order*max_dim*Core_max);
	f_Resp = (double *)malloc(sizeof(double)*order*max_dim*Core_max);
    // initilize factor matrix and indexings
    // something more efficient will be better 
    for (int i = 0; i < order; i++) {
		int row = dimensionality[i], col = Core_size[i];
		if(!AUTO) target_FM_N[i] = pratio*row*col+1;  
        Pruned_FM_N[i] = 0;
		int s_index = 0;
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				FactorM[i*max_dim*Core_max + j*Core_max + k] = frand(0, 1);
				//FactorM[i*max_dim*Core_max + j*Core_max + k] = Gaussian();
				markFM[i*max_dim*Core_max + j*Core_max + k] = 0;
				f_Num[i*max_dim*Core_max + s_index] = i*max_dim*Core_max + j*Core_max + k;
				s_index++;
			}
		}
	}
    // Initialize output core tensor parameters and data structures
	CoreTensor = (double *)malloc(sizeof(double)*Core_N);
	CorePermu = (int *)malloc(sizeof(int)*Core_N*order);
	markCore = (bool*)malloc(sizeof(int)*Core_N);
	c_Resp = (double*)malloc(sizeof(double)*Core_N);
	c_Num = (int*)malloc(sizeof(int)*Core_N);
	if(!AUTO) target_Core_N = Core_N*pratio + 1; 
    int pos = 0;
	for (int i = 0; i < Core_N; i++) {
		CoreTensor[i] = frand(0, 1);
		//CoreTensor[i] = Gaussian();
		markCore[i] = 0;
		c_Num[i] = i;
		if (i == 0) {
			for (int j = 0; j < order; j++) CorePermu[j] = 0;
		} 
        else {
			for (int j = 0; j < order; j++) {
				CorePermu[i*order + j] = CorePermu[(i - 1)*order + j];
			}
			CorePermu[i*order + order - 1]++;  
            int k = order - 1;
			while (CorePermu[i*order + k] >= Core_size[k]) {
				CorePermu[i*order + k] -= Core_size[k];
				CorePermu[i*order + k - 1]++; k--;
			}
		}
	}
    // size adjusted LAMBDA value 
    if(!FIXED_LAMBDA)
	    LAMBDA *= ((double)Entries_N)/((double)Core_N+(double)t_FM_N);
    
}   

// FOR TEST ONLY - read in fixed initial FMs and Core 
void Initialize_read_init(){
    if(VERBOSE) printf("Initialize...\n"); 
	// Initialize output factor matrix intermediate data structures
    assign_index();
    FactorM = (double *)malloc(sizeof(double)*order*max_dim*Core_max);
    f_Num = (int *)malloc(sizeof(int)*order*max_dim*Core_max);
	if(!AUTO) target_FM_N = (int *)malloc(sizeof(int)*order);
    Pruned_FM_N = (int *)malloc(sizeof(int)*order);
	markFM = (bool *)malloc(sizeof(int)*order*max_dim*Core_max);
	f_Resp = (double *)malloc(sizeof(double)*order*max_dim*Core_max);
    
    // initilize factor matrix and indexings
    // something more sfficient will be better 
 
    char fmFN[1024]; 
    strcpy(fmFN,InputPath); strcat(fmFN, "_FM");   
    FILE *FM = fopen(fmFN, "r");
    bool is_there = true; 
    if(FM == NULL){
        is_there = false;
        FM = fopen(fmFN, "w"); 
        if(VERBOSE) printf("INIT FM new\n");
    }else{
        if(VERBOSE) printf("INIT FM old\n");
    }
    for (int i = 0; i < order; i++) {
		int row = dimensionality[i], col = Core_size[i];
		if(!AUTO) target_FM_N[i] = pratio*row*col+1;  
        Pruned_FM_N[i] = 0;
		int s_index = 0;
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				if(is_there){ // read in 
                    char x[100];  
                    if(fscanf(FM, "%100s", x) == 1) 
                        FactorM[i*max_dim*Core_max + j*Core_max + k] = atof(x);
                    else
                        FactorM[i*max_dim*Core_max + j*Core_max + k] = 0.0;
                }else{ // write and store
                    double randNum = frand(0,1); // Gaussina();
                    FactorM[i*max_dim*Core_max + j*Core_max + k] = randNum;
                    fprintf(FM, "%f\t", randNum); 
                }
				markFM[i*max_dim*Core_max + j*Core_max + k] = 0;
				f_Num[i*max_dim*Core_max + s_index] = i*max_dim*Core_max + j*Core_max + k;
				s_index++;
			}
            fprintf(FM, "\n"); 
		}
    }
    fclose(FM); 

    // Initialize output core tensor parameters and data structures
	CoreTensor = (double *)malloc(sizeof(double)*Core_N);
	CorePermu = (int *)malloc(sizeof(int)*Core_N*order);
    markCore = (bool*)malloc(sizeof(int)*Core_N);
	c_Resp = (double*)malloc(sizeof(double)*Core_N);
	c_Num = (int*)malloc(sizeof(int)*Core_N);
	if(!AUTO) target_Core_N = Core_N*pratio + 1; 
    int pos = 0;

    char cfFN[1024]; 
    strcpy(cfFN,InputPath); strcat(cfFN, "_Core"); 
    FILE *CF = fopen(cfFN, "r");
    is_there = true; 
    if(CF == NULL){
        is_there = false;
        CF = fopen(cfFN, "w"); 
        if(VERBOSE) printf("INIT CORE new\n");
    }else{
        if(VERBOSE) printf("INIT CORE old\n");
    }

    for (int i = 0; i < Core_N; i++) {
	    if(is_there){ // read in 
            char x[100];  
            if(fscanf(CF, "%100s", x) == 1) 
    	        CoreTensor[i] = atof(x);
            else
                CoreTensor[i] = 0; 
        }else{
            double randNum= frand(0, 1);
        	CoreTensor[i] = randNum;
            fprintf(CF, "%f\t", randNum); 
		}
        
        markCore[i] = 0;
		c_Num[i] = i;
		if (i == 0) {
			for (int j = 0; j < order; j++) CorePermu[j] = 0;
		} 
        else {
			for (int j = 0; j < order; j++) {
				CorePermu[i*order + j] = CorePermu[(i - 1)*order + j];
			}
			CorePermu[i*order + order - 1]++;  
            int k = order - 1;
			while (CorePermu[i*order + k] >= Core_size[k]) {
				CorePermu[i*order + k] -= Core_size[k];
				CorePermu[i*order + k - 1]++; k--;
			}
		}
	}
    fclose(CF);

    // normalized LAMBDA value 
	LAMBDA *= ((double)Entries_N)/((double)Core_N+(double)t_FM_N);
    
}   


//[Input] Input tensor X, initialized or updated factor matrices A^{(n)} (n=1,...,N) and core tensor G
//[Output] Updated core tensor G
//[Function] Update all core tensor entries by a entry-wise update rule derived from L_1 regularized loss function
void Update_Core_Tensor() {

    double* Save_1 = (double *)malloc(sizeof(double)*Entries_N);

	int mull = max_dim*Core_max;
#pragma omp parallel for schedule(static) 
	for (int i = 0; i < Entries_N; i++){
		double *cach = (double *) malloc(sizeof(double)*order);
		//double *cach = new double[order];
		for (int j = 0; j < order; j++) cach[j] = Index[i*order + j];
		double ans=0;
		for (int j = 0; j < Core_N; j++){
			double temp = CoreTensor[j];
			for (int k = 0; k < order; k++){
				int mulrow = cach[k];
                int mulcol = CorePermu[j*order + k];
				temp *= FactorM[k*mull+mulrow*Core_max+mulcol];
			}
			ans+=temp;
		}
		Save_1[i] = ans;
		free(cach);
	}
	//Cannot be parallelized
	for (int i = 0; i < Core_N; i++) {
		if(markCore[i] != 0) continue; 
		double *cach = (double *) malloc(sizeof(double)*order);
		double g = 0;
		double d = 0;
		for (int j = 0; j < order; j++) cach[j] = CorePermu[i*order + j];
		for (int j = 0; j < Entries_N; j++) {
			double temp = 1;
			for (int k = 0; k < order; k++) {
				int mulrow = Index[j*order + k];
                int mulcol = cach[k];
				temp *= FactorM[k*mull + mulrow*Core_max + mulcol];
			}
			if(loss_type=='1'){ // L1 loss 
                g += (-2)*(Entries[j]-Save_1[j]+(temp*CoreTensor[i]))*temp;
			    d += 2*temp * temp;		
            }else{  // LF loss
                g += (Entries[j]-Save_1[j]+(temp*CoreTensor[i]))*temp;
			    d += temp * temp;		
            }
			Save_1[j] -= CoreTensor[i]*temp;	
		}

	    if(loss_type == '1'){   // L1 loss
            if(g > LAMBDA) CoreTensor[i] = (LAMBDA - g)/d;
		    else if (g < -LAMBDA) CoreTensor[i] = -(LAMBDA+g)/d;
		    else CoreTensor[i] = 0;
        }else{
	        CoreTensor[i] = g/(d + LAMBDA);
        }
		for (int j = 0; j < Entries_N; j++) {
			double temp=1;
			for (int k = 0; k < order; k++) {
				int mulrow = Index[j*order + k], mulcol = cach[k];
				temp *= FactorM[k*mull + mulrow*Core_max + mulcol];
			}
			Save_1[j] += CoreTensor[i]*temp;	
		}	
        free(cach);
	}
	free(Save_1);
}


//[Input] Input tensor X, initialized core tensor G, and factor matrices A^{(n)} (n=1...N)  
//[Output] Updated factor matrices A^{(n)} (n=1...N)
//[Function] Update all factor matrice entries by a entry-wise update rule derived from L_1 regularized loss function
void Update_Factor_Matrices() {

	int mult = max_dim*Core_max;
	for (int i = 0; i < order; i++) { //Updating the ith Factor Matrix
		int row_N = dimensionality[i];
		int column_N = Core_size[i];
		
		// store index of all nonzero elements of core-tensor 
        int fmsize = Core_N;
        if(MARK) fmsize -= Pruned_Core_N; 
		int *nz_core_ind = (int*)malloc(sizeof(int)*(fmsize));
		int nnzc = 0; //number of nonzero entries of core tensor
		for (int l = 0; l < Core_N; l++){
			if(MARK) if(CoreTensor[l] == 0) continue;
			nz_core_ind[nnzc] = l;
			nnzc++;
		}

#pragma omp parallel for schedule(static) //in parallel
		for (int j = 0; j < row_N; j++) {
			for(int k = 0; k < column_N; k++){
				if(markFM[i*mult + j*Core_max + k] != 0) continue; 

				double *Delta = (double *)malloc(sizeof(double)*column_N);
				double *V = (double *)malloc(sizeof(double)*column_N);

				double e = 0;
				//Initialize V
				for (int l = 0; l < column_N; l++) {
					V[l] = 0;
				}

				int pos = i*max_dim + j;
				int nnz = CountX[pos + 1] - CountX[pos];
				pos = CountX[pos];
				for (int l = 0; l < nnz; l++) { //Updating Delta and V
					int current_input_entry = WhereX[pos + l];
					int pre_val = current_input_entry*order;
					int *cach1 = (int *)malloc(sizeof(int)*order);
					for (int ll = 0; ll < order; ll++) cach1[ll] = Index[pre_val++];
					for (int ll = 0; ll < column_N; ll++) Delta[ll] = 0;
					for (int ll = 0; ll < nnzc; ll++) {
						int nzidx = nz_core_ind[ll];
						int pre1 = nzidx*order, pre2 = 0;
						int CorePos = CorePermu[pre1 + i];
						double res = CoreTensor[nzidx];
						for (int ii = 0; ii < order; ii++) {
							if (ii != i) {
								int mulrow = cach1[ii], mulcol = CorePermu[pre1];
								res *= FactorM[pre2 + mulrow*Core_max + mulcol];
							}
							pre1++;
							pre2 += mult;
						}
						Delta[CorePos] += res;
					}
					free(cach1);
					int now = 0;
					double Entry_val = Entries[current_input_entry];

					double cach = Delta[k];
					for (int ii = 0; ii < column_N; ii++) {
						V[ii] += cach * Delta[ii];
					}
					e += cach * Entry_val;
				}
				free(Delta);

				//Update the (j,k) entries of ith Factor Matrix 
				int cach = i*mult + j*Core_max;
				double res = 0;
				for (int l = 0; l < column_N; l++) {
					if(l == k) continue;
					res += V[l] * FactorM[cach+l];
				}
				if(loss_type == '1'){
                    double g = (-2)*(e-res);
				    double d = 2*V[k];
				    if(g > LAMBDA) {
					    FactorM[cach + k] = (LAMBDA - g)/d;
				    }
			    	else if(g < LAMBDA*-1) {
		    			FactorM[cach + k] = -(LAMBDA+g)/d;
	    			}
    				else {
					    FactorM[cach + k] = 0;
				    }
                }else{ // loss type LF
        			FactorM[cach + k] = (e-res)/(V[k] + LAMBDA);
                }

				free(V);
			}
		}
        free(nz_core_ind);
	}
}


//[Input] Input tensor X, core tensor G, and factor matrices A^{(n)} (n=1...N)
//[Output] Fit = 1-||X-X'||/||X|| (Reconstruction error = ||X-X'||)
//[Function] Calculating fit and reconstruction error in a parallel way.
void Reconstruction() {
	RE = RMSE = Error = 0;
	double* Error_T = (double *)malloc(sizeof(double)*Entries_N);

#pragma omp parallel for schedule(static)
	for (int i = 0; i < Entries_N; i++) {
		Error_T[i] = Entries[i];
	}
	int mult = max_dim*Core_max;
#pragma omp parallel for schedule(static)
	for (int i = 0; i < Entries_N; i++) {
		int j, pre_val = i*order;
		double ans = 0;
		int *cach1 = (int *)malloc(sizeof(int)*order);
		//int *cach1 = new int[order];
		for (int j = 0; j < order; j++) cach1[j] = Index[pre_val++];
		for (int j = 0; j < Core_N; j++) {
			double temp = CoreTensor[j];
			int pos = j*order;
			int val = 0;
			for (int k = 0; k < order; k++) {
				int mulrow = cach1[k], mulcol = CorePermu[pos++];
				temp *= FactorM[val + mulrow*Core_max + mulcol];
				val += mult;
			}
			ans += temp;
		}
		free(cach1);
		Error_T[i] -= ans;
	}

#pragma omp parallel for schedule(static) reduction(+:Error)
	for (int i = 0; i < Entries_N; i++) {
		Error += Error_T[i] * Error_T[i];
	}
	if(NormX != 0) RE = sqrt(Error)/NormX;
	else printf("Problem Occured; check input tensor\n");
	RMSE = sqrt((RE*RE)/Entries_N);
	free(Error_T);
}


//[Input] Input tensor X, updated factor matrices A^{(n)} (n=1,...,N) and core tensor G, and marking table
//[Output] Pruned core tensor G, updated marking table
//[Function] Calculating Resp(G_gamma) for core tensor entry G_gamma and prune core tensor entries
void Calculate_Core_Resp() {
	double* tc_Error_T = (double *) malloc(sizeof(double)*Entries_N);
	int unmarked_Core_N = Core_N;
    if(MARK) unmarked_Core_N -= Pruned_Core_N;

#pragma omp parallel for schedule(static)
	for (int i=0; i<Core_N; i++) {
		c_Resp[i] = 0;
	}
#pragma omp parallel  for schedule(static)
	for (int i = 0; i < Entries_N; i++){
		tc_Error_T[i] = Entries[i];
	}
	int mull = max_dim*Core_max;
#pragma omp parallel for schedule(static)
	for (int i = 0; i < Entries_N; i++){
		double *cach = (double *) malloc(sizeof(double)*order);
		for (int j = 0; j < order; j++) cach[j] = Index[i*order + j];
		double ans=0;
		for (int j = 0; j < Core_N; j++){
			double temp = CoreTensor[j];
			for (int k = 0; k < order; k++){
				int mulrow = cach[k], mulcol = CorePermu[j*order + k];
				temp *= FactorM[k*mull+mulrow*Core_max+mulcol];
			}
			ans+=temp;
		}
		free(cach);
		tc_Error_T[i] -= ans;
	}
#pragma omp parallel for schedule(static)
	for (int i = 0; i < unmarked_Core_N; i++) {
		int idx = c_Num[i];
		double *cach = (double *) malloc(sizeof(double)*order);
		double ans = 0;
		for (int j = 0; j < order; j++) cach[j] = CorePermu[idx*order + j];
		for (int j = 0; j < Entries_N; j++) {
			double temp = CoreTensor[idx];
			for (int k = 0; k < order; k++) {
				int mulrow = Index[j*order + k], mulcol = cach[k];
				temp *= FactorM[k*mull + mulrow*Core_max + mulcol];
			}
			ans += (tc_Error_T[j]+temp)*(tc_Error_T[j]+temp);
		}	
		free(cach);

		c_Resp[idx] = ans;
	}
	std::sort(c_Num, c_Num + unmarked_Core_N, comp_c);


	free(tc_Error_T);
}


//[Input] Input tensor X, updated factor matrices A^{(n)} (n=1,...,N) and core tensor G, and marking table
//[Output] Pruned factor matrix A^{(i)} and updated marking table
//[Function] Calculating Resp(a^{i}_{jk}) for factor matrix entry a^{i}_{jk} and prune factor matrix entries
void Calculate_FM_Resp() {

	for(int i=0; i<order; i++){
		int row = dimensionality[i], col = Core_size[i];
		int mull = max_dim*Core_max;
		int unmarked_FM_N = row*col;
        if(MARK) unmarked_FM_N -= Pruned_FM_N[i];
		
		//printf("check\n");
		for (int j = 0; j < row; j++) {
			for (int k = 0; k < col; k++) {
				int temp = i*Core_max*max_dim + j*Core_max + k;
				f_Resp[temp] = 0;
			}
		}

//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for schedule(static)
	for(int s_index = 0; s_index < unmarked_FM_N; s_index++){
		   int idx = f_Num[i*Core_max*max_dim + s_index];
		   int j_index = (idx-i*Core_max*max_dim)/Core_max;
		   int k_index = (idx-i*Core_max*max_dim)%Core_max;

				int pos = i*max_dim + j_index;
				int e_Entries_N = CountX[pos+1] - CountX[pos];

				double* n_Error_T;
				n_Error_T = (double *)malloc(sizeof(double)*e_Entries_N);
				int e_Core_N = 1;
				int* tempcoreidx;
				for (int q = 0; q < order; q++){
					if(q==i) continue;
					e_Core_N *= Core_size[q];
				}
				tempcoreidx = (int*)malloc(sizeof(int)*e_Core_N);
				int n_i = 0;
				for (int q = 0; q < Core_N; q++){
					if(CorePermu[order*q+i] == k_index){
						tempcoreidx[n_i] = q;
						n_i++;
					}
					if(n_i == e_Core_N) break;
				}

				pos = CountX[pos];
				for (int n_i = 0; n_i < e_Entries_N; n_i++){
					int current_input_entry = WhereX[pos+n_i];
					int pre_val = current_input_entry*order;
					//int n_j;
					double *cach = (double *) malloc(sizeof(double)*order);
					for (int n_j = 0; n_j < order; n_j++) {
						cach[n_j] = Index[pre_val++];
					}
					double ans1=0;
					double ans2=0;
					int tmp_idx=0;
					for (int n_j = 0; n_j < Core_N; n_j++){
						double n_temp = CoreTensor[n_j];
						for (int n_k = 0; n_k < order; n_k++){
							int mulrow = cach[n_k];
							int mulcol = CorePermu[n_j*order + n_k];
							n_temp *= FactorM[n_k*mull + mulrow*Core_max + mulcol];
						}
						ans1+=n_temp;
						if(n_j == tempcoreidx[tmp_idx]){
							ans2 += n_temp;
							if(tmp_idx < e_Core_N) tmp_idx++;
						}
					}
					free(cach);
					if(tmp_idx != e_Core_N) {
						cout << "problem happens" << endl;
						cout << "(i, j, k) = (" << i << ", " << j_index << ", " << k_index << "): ";
						cout << tmp_idx << "(e_Core_N = " << e_Core_N << ")" << endl;
					}
					n_Error_T[n_i] = (2*Entries[current_input_entry] - 2*ans1 + ans2)*ans2;
				}

				double f_Error = 0;
				for(int n_i = 0; n_i<e_Entries_N; n_i++){
					f_Error += n_Error_T[n_i];
				}

				f_Resp[idx] = f_Error;

				free(n_Error_T);
				free(tempcoreidx);
			}
		int start = i*Core_max*max_dim;
		sort(f_Num + start, f_Num + start + unmarked_FM_N, comp_f);

	}
}


//[Input] Input tensor X, updated factor matrices A^{(n)} (n=1,...,N) and core tensor G, and marking table
//[Output] Pruned factor matrix A^{(i)} and core tensor G, and updated marking table 
//[Function] Calculating Resp value and prune elements
double Pruning(int g_iter){
    int Remove;
    double pR = MIN(INIT_PR*g_iter, MAX_PR); 
    int countPR = 0;
	//double pR = 0.1;
	Calculate_Core_Resp();
	
    // Prune Core Tensor
	int unmarked_Core_N = Core_N - Pruned_Core_N;
    if(AUTO){
        Remove = pR*Core_N;
        if(unmarked_Core_N - Remove < Core_max) {
            Remove = MAX(unmarked_Core_N - Core_max,0);
            countPR++;  
        }
    }else{
        if(target_Core_N - Pruned_Core_N > pR*Core_N) 
            Remove = pR*Core_N;
	    else{ 
            Remove = target_Core_N - Pruned_Core_N;
            countPR++;
        }
    }
   
    if(MARK){ 
        for(int i=0; i < Remove; i++){
		    int now = c_Num[unmarked_Core_N - 1 - i];
            CoreTensor[now] = 0;
		    markCore[now] = 1;
	    }
	}else{
        for(int i=0; i < Remove + Pruned_Core_N; i++){
		    int now = c_Num[Core_N - 1 - i];
            CoreTensor[now] = 0;
	    }	
    }
    Pruned_Core_N += Remove;
    
    // calculate FM responsibility
    Calculate_FM_Resp();

	// Prune Factor Matrices
	for(int i=0; i<order; i++){
        int unmarked_FM_N = FM_N[i] - Pruned_FM_N[i];
		if(AUTO){
            Remove = pR*FM_N[i];
            if(unmarked_FM_N - Remove  < dimensionality[i]){
                Remove = MAX(unmarked_FM_N - dimensionality[i],0); 
                countPR++; 
            }
        }else{
            if(target_FM_N[i] - Pruned_FM_N[i] > pR*FM_N[i]) Remove = pR*FM_N[i];
		    else{ 
                Remove = target_FM_N[i] - Pruned_FM_N[i];
                countPR++;
            }
        }
        
        if(MARK){
		    for(int j=0; j < Remove; j++){
			    int now = f_Num[i*Core_max*max_dim + unmarked_FM_N - 1 -j];
			    int row_idx = (now-i*Core_max*max_dim)/Core_max;
			    int col_idx = (now-i*Core_max*max_dim)%Core_max;
			    if(row_idx > dimensionality[i]) cout << "Problem occured, rowidx:" << row_idx << " > row:" << dimensionality[i] << endl;
			    if(col_idx > Core_size[i]) cout << "Problem occured, colidx:" << col_idx << " > col:" << Core_size[i] << endl;

		    	FactorM[now] = 0;
			    if(MARK) markFM[now] = 1;
		    }
        }else{
            for(int j=0; j < Remove+Pruned_FM_N[i]; j++){
			    int now = f_Num[i*Core_max*max_dim + FM_N[i] - 1 -j];
			    int row_idx = (now-i*Core_max*max_dim)/Core_max;
			    int col_idx = (now-i*Core_max*max_dim)%Core_max;
			    if(row_idx > dimensionality[i]) cout << "Problem occured, rowidx:" << row_idx << " > row:" << dimensionality[i] << endl;
			    if(col_idx > Core_size[i]) cout << "Problem occured, colidx:" << col_idx << " > col:" << Core_size[i] << endl;

		    	FactorM[now] = 0;
		    }
        }
		Pruned_FM_N[i] += Remove;
	}
    
    if(countPR == order+1) keepPruning = false; 

	int t_pruned = Pruned_Core_N;
	int t_N = Core_N;
	for(int i = 0; i < order; i++){
		t_pruned += Pruned_FM_N[i];
		t_N += FM_N[i];
	}
	totalsparsity = (double)t_pruned/(double)t_N;

//    return;
    return pR; 
}


// Purning stopping criterion 
void CheckOP(int g_iter, double pR){
	double avgd;
    if(keepPruning == false) return;
    if(g_iter==1)
        RE_max = RE_min = RE;
    if(pR < 0.09){
	     double dd = RE + preREs[0] -2.0*preREs[1];  	
        if(VERBOSE){ printf("2nd derivative calculated:%.3f + %.3f -2.0*%.3f = %.3f\n", RE, preREs[0], preREs[1], dd); }
        if(RE > RE_max) RE_max = RE;
        if(RE < RE_min) RE_min = RE;
        // update previous REs
        preREs[0] = preREs[1];
        preREs[1] = RE;
    } else {
        // estimate the 2nd derivative dd to determin the elbow point  
	    double dd = RE + preREs[0] - 2.0*preREs[1];  	
        if(VERBOSE){ printf("2nd derivative calculated:%.3f + %.3f -2.0*%.3f = %.3f\n", RE, preREs[0], preREs[1], dd); }
        if( dd > DELTA && RE-preREs[2] > STOP_RE_DIFF*20.0){
            keepPruning = false;
			return;
	    }
        // for cases without a elbow point (linely increasing REs)
        if( RE>=RE_min*(1.0 + STOP_RE_P) ){ 
            keepPruning = false;
			return;
	    }

        // Update 
		if(RE > RE_max) RE_max = RE;
        if(RE < RE_min) RE_min = RE;
        preREs[0] = preREs[1]; 
        preREs[1] = RE;
    }
    return; 
}

void RevivePE(double pR){
	int Revive;

	// Revive Core Tensor
	Revive = pR*Core_N;
	int unmarked_Core_N = Core_N - Pruned_Core_N;
	for(int i=0; i < Revive; i++){
		int now = c_Num[unmarked_Core_N + i];
	    markCore[now] = 0;
	}
	Pruned_Core_N -= Revive;

	// Prune Factor Matrices
	for(int i=0; i<order; i++){
		Revive = pR*FM_N[i];
		int unmarked_FM_N = FM_N[i] - Pruned_FM_N[i];
		for(int j=0; j < Revive; j++){
			int now = f_Num[i*Core_max*max_dim + unmarked_FM_N + j];
			int row_idx = (now-i*Core_max*max_dim)/Core_max;
			int col_idx = (now-i*Core_max*max_dim)%Core_max;
			if(row_idx > dimensionality[i]) cout << "Problem occured, rowidx:" << row_idx << " > row:" << dimensionality[i] << endl;
			if(col_idx > Core_size[i]) cout << "Problem occured, colidx:" << col_idx << " > col:" << Core_size[i] << endl;

			markFM[now] = 0;
		}
		Pruned_FM_N[i] -= Revive;
	}
	
	int t_pruned = Pruned_Core_N;
	int t_N = Core_N;
	for(int i = 0; i < order; i++){
		t_pruned += Pruned_FM_N[i];
		t_N += FM_N[i];
	}
	totalsparsity = (double)t_pruned/(double)t_N;

	alreadyRevived = true;
}


//[Input] Updated factor matrices A^{(n)} (n=1...N)
//[Output] Standardized factor matrices A^{(n)} (n=1...N) and tuned core tensor G
//[Function] Standardize all column of factor matrices and update core tensor simultaneously.
void Standardize() {
	Mul = (int*)malloc(sizeof(int)*order);
	Mul[order - 1] = 1;
	for (int i = order - 2; i >= 0; i--) {
		Mul[i] = Mul[i + 1] * Core_size[i + 1];
	}
	int pos = 0;
	for (int i = 0; i < order; i++) {
		mat X = mat(dimensionality[i], Core_size[i]);
		mat N = mat(Core_size[i], Core_size[i]);
		double* normval = (double*)malloc(sizeof(double)*Core_size[i]);
		N.zeros();
		for(int j = 0; j < Core_size[i]; j++){
			normval[j] = 0;
		}
		for (int j = 0; j < dimensionality[i]; j++) {
			for (int k = 0; k < Core_size[i]; k++) {
				X(j, k) = FactorM[i*max_dim*Core_max + j*Core_max + k];
				normval[k] += FactorM[i*max_dim*Core_max + j*Core_max + k]*FactorM[i*max_dim*Core_max + j*Core_max + k];
			}
		}
		for(int j = 0; j < Core_size[i]; j++){
			N(j,j) = sqrt(normval[j]);
		}
		mat Y = normalise(X);
		for (int j = 0; j < dimensionality[i]; j++) {
			for (int k = 0; k < Core_size[i]; k++) {
				FactorM[i*max_dim*Core_max + j*Core_max + k] = Y(j, k);
			}
		}

		tempCore = (double*)malloc(sizeof(double)*Core_N);
		tempPermu = (int*)malloc(sizeof(int)*order);
		for (int j = 0; j < Core_N; j++) {
			tempCore[j] = 0;
		}
		for (int j = 0; j < Core_N; j++) {
			for (int k = 0; k <= i - 1; k++) {
				tempPermu[k] = CorePermu[j*order + k];
			}
			for (int k = i + 1; k < order; k++) {
				tempPermu[k] = CorePermu[j*order + k];
			}
			for (int k = 0; k < Core_size[i]; k++) {
				tempPermu[i] = k;
				int cur = j + (k - CorePermu[j*order + i])*Mul[i];
				tempCore[cur] += CoreTensor[j] * N(k, CorePermu[j*order + i]);
			}
		}
		for (int j = 0; j < Core_N; j++) {
			CoreTensor[j] = tempCore[j];
		}
		free(normval);
	}
	free(Mul);
	free(tempCore);
	free(tempPermu);
}


//[Input] test set tensor Y, updated core tensor G, and updated factor matrices A^{(n)} (n=1...N)
//[Output] Test RMSE  = sqrt(||Y-Y'||/||Y||)
//[Function] Calculating Test RMSE in a parallel way.
double Test() {
	TestRE = Error = 0;
	double testNormSq = 0;
	double* Error_T = (double *)malloc(sizeof(double)*Entries_N);
#pragma omp parallel for schedule(static)
	for (int i = 0; i < test_N; i++) {
		Error_T[i] = 0;
	}
	int mult = max_dim*Core_max;
#pragma omp parallel for schedule(static)
	for (int i = 0; i < test_N; i++) {
		// int j; 
        int pre_val = i*order;
		double ans = 0;
		int *cach1 = (int *)malloc(sizeof(int)*order);
		for (int j = 0; j < order; j++) cach1[j] = I2[pre_val++];
		for (int j = 0; j < Core_N; j++) {
			double temp = CoreTensor[j];
			// int k;
			int pos = j*order;
			for (int k = 0; k < order; k++) {
				int mulrow = cach1[k], mulcol = CorePermu[pos++];
				temp *= FactorM[k*mult + mulrow*Core_max + mulcol];
			}
			ans += temp;
		}
		free(cach1);
		Error_T[i] += ans;
	}
#pragma omp parallel for schedule(static)
	for (int i = 0; i < test_N; i++) {
		double est = Error_T[i];
		Error += (E2[i] - est) * (E2[i] - est);
		testNormSq += E2[i]*E2[i];
	}
	TestRE = sqrt(Error/testNormSq);

	if(VERBOSE) printf("\nTest RE:\t%lf\nMAE:\t%lf\n", TestRE, MAE);

	free(Error_T);
    return (TestRE);
}



//[Input] Input tensor X, initialized core tensor G, and initialized factor matrices A^{(n)} (n=1...N)
//[Output] Updated core tensor G and factor matrices A^{(n)} (n=1...N)
//[Function] Performing main algorithm which updates core tensor and factor matrices iteratively
double Vest() {
//	printf("Starting VeST.\n");

    double Stime = omp_get_wtime();         // start time
    double avertime = 0;                    // avg time 
    int g_iter = 0;                         // iteration cound 
	double pR = INIT_PR;

	while (g_iter <= MAX_ITER) {
		double itertime = omp_get_wtime();
		double steptime;
        g_iter++; 
		if(VERBOSE) printf("\n[Iteration %d]\n", g_iter);
		steptime = omp_get_wtime();
        Update_Factor_Matrices(); 
		Update_Core_Tensor();	
		if(VERBOSE) printf("Elapsed time for updating elements:\t%lf\n", omp_get_wtime() - steptime);

		steptime = omp_get_wtime();
		Reconstruction();
		if(VERBOSE) printf("Elapsed time for calculating RE %f:\t%lf\n", RE, omp_get_wtime() - steptime);

	    // pR = CheckOP(g_iter, pR); 
        if(AUTO){
            CheckOP(g_iter,pR);
            if(keepPruning){
            // if(pR>=INIT_PR){
			    steptime = omp_get_wtime();
			    pR = Pruning(g_iter);
			    if(VERBOSE) printf("Elapsed time for pruning elements:\t%lf\n", omp_get_wtime() - steptime);
            }
		    else if(!alreadyRevived){
			    steptime = omp_get_wtime();
			    if(VERBOSE) printf("Stop pruning & Revive elements\n");
			    RevivePE(pR);
			    if(VERBOSE) printf("Elapsed Time for Reviving elements:\t%lf\n", omp_get_wtime() - steptime);
		    }
        } else if(!AUTO && pratio>0.0){ // manual with pratio given 
		    if(totalsparsity < pratio){
			    steptime = omp_get_wtime();
			    pR = Pruning(g_iter);
			    if(VERBOSE) printf("Elapsed Time for Pruning elements:\t%lf\n", omp_get_wtime() - steptime);
		    }
        }
	int count_zero = 0; 	
	int total_zero = 0;
	for (int i = 0; i < order; i++) {
		for (int j = 0; j < dimensionality[i]; j++) {
			for (int k = 0; k < Core_size[i]; k++) {
				if(FactorM[i*mult + j*Core_max + k] == 0) count_zero++;
			}
		}
	}
    
    	double zero_rat = (double)count_zero/(double)t_FM_N;
	fZR = zero_rat; 

	total_zero += count_zero;
	count_zero=0;
	pos = 0;
	for (int i = 0; i < Core_N; i++) {
		if(CoreTensor[i] == 0) {
			count_zero++;
			continue;	
		}
	}

	zero_rat = (double)count_zero/(double)Core_N;
	total_zero += count_zero;
	double ZR = (double)total_zero/(double)(t_FM_N+Core_N);
	totalsparsity = ZR;

		if(VERBOSE) printf("\nSparsity:\t%lf\tRE:\t%lf\tElapsed Time:\t%lf\n\n", totalsparsity, RE, omp_get_wtime() - itertime);
		avertime += omp_get_wtime() - itertime;
        
        // loop stopping criterion for auto mode
		if (AUTO && !keepPruning && pRE != -1 && pRE - RE <= STOP_RE_DIFF && pRE-RE >= 0) {
			if(VERBOSE) printf("pRE=%lf\tRE=%lf\tpRE-RE=%lf\n",pRE,RE,pRE-RE);
			break;
		}
        // stopping criteria for manual mode
        if (!AUTO && totalsparsity >= pratio && pRE != -1 && pRE - RE <= STOP_RE_DIFF && pRE-RE >= 0) {
			printf("pRE=%lf\tRE=%lf\tpRE-RE=%lf\n",pRE,RE,pRE-RE);
			break;
		}
        // stop if too sparse
        if(totalsparsity>=1.0){
            break;
        }
		pRE = RE;
	}

    iterNum = g_iter; 
	avertime /= g_iter;
	avgITime = avertime; 

    if(VERBOSE) printf("\nIteration ended.\tRE : %lf\tAverage iteration time : %lf\n", RE, avertime);

	if(VERBOSE) printf("\nStandardizing...\n");
	Standardize();

    if(TESTING){
	    if(VERBOSE) printf("\nCalculate test RMSE...\n");
	    Test();	
    }
	tTime = omp_get_wtime() - Stime;
    if(VERBOSE) printf("\nUPDATES DONE!\tFinal RE : %lf\tTotal Elapsed time: %lf\n\n", RE,tTime);
    
    return RE;
}


//[Input] Updated core tensor G and factor matrices A^{(n)} (n=1...N)
//[Output] core tensor G in sparse tensor format and factor matrices A^{(n)} (n=1...N) in full-dense matrix format(truncated entries marked as 0)
//[Function] Writing all factor matrices and core tensor in result path
void Print() {
	if(VERBOSE) printf("\nWriting factor matrices and the core tensor...\n");
	char temp[50];
	int pos = 0;
	int count_zero = 0; 	
	int total_zero = 0;
	int mult = max_dim*Core_max;
	for (int i = 0; i < order; i++) {
		sprintf(temp, "%s/FACTOR%d", ResultPath, i);
		FILE *fin = fopen(temp, "w");
		for (int j = 0; j < dimensionality[i]; j++) {
			for (int k = 0; k < Core_size[i]; k++) {
				if(fin!=NULL) fprintf(fin, "%e\t", FactorM[i*mult + j*Core_max + k]);
				if(FactorM[i*mult + j*Core_max + k] == 0) count_zero++;
			}
			if(fin!=NULL) fprintf(fin, "\n");
		}
        fclose(fin);
	}
    
    double zero_rat = (double)count_zero/(double)t_FM_N;
	if(VERBOSE){ 
        printf("\n---FactorMatrix---");
	    printf("\nnumber of zero entry is %d and nonzero entry is %d", count_zero, t_FM_N-count_zero);
	    printf("\nnumber of zero entry ratio is %lf...\n", zero_rat);
    }
    fZR = zero_rat; 

	total_zero += count_zero;
	count_zero=0;
	sprintf(temp, "%s/CORETENSOR", ResultPath);
	FILE *fcore = fopen(temp, "w");
	pos = 0;
	for (int i = 0; i < Core_N; i++) {
		if(CoreTensor[i] == 0) {
			count_zero++;
			continue;	
		}
        if(fcore!=NULL){ 
		    for (int j = 0; j < order; j++) {
			    fprintf(fcore, "%d\t", CorePermu[pos++]);
		    }
		    fprintf(fcore, "%e\n", CoreTensor[i]);
        }
	}
    fclose(fcore);

	zero_rat = (double)count_zero/(double)Core_N;
	total_zero += count_zero;
	double ZR = (double)total_zero/(double)(t_FM_N+Core_N);
	totalsparsity = ZR;
	if(VERBOSE){
        printf("\n---CoreTensor---");
	    printf("\nnumber of zero entry is %d and nonzero entry is %d", count_zero, Core_N-count_zero);
	    printf("\nnumber of zero entry ratio is %lf...\n", zero_rat);
	    printf("\nnumber of total zero entry ratio is %lf...\n", ZR);
    }
    cZR = zero_rat;
    tZR = ZR; 
    return; 
}


//[Input] Path of input tensor file, result directory, tensor order, tensor rank, and number of threads
//[Output] Core tensor G and factor matrices A^{(n)} (n=1,...,N)
//[Function] Performing VEST for a given sparse tensor
int main(int argc, char* argv[]) {

	if (argc >= 5+atoi(argv[4])) {
        // Must specified arguments 
		InputPath = argv[1];
		if(strcmp(argv[2], "NA")!=0){ 
            testInputPath = argv[2];
            TESTING = true; 
        }
		ResultPath = argv[3];
		order = atoi(argv[4]);

		Core_size = (int*)malloc(sizeof(int)*order);
		for(int i=0; i<order; i++){
			Core_size[i] = atoi(argv[5+i]);
            // if(Core_size[i] > Core_max) Core_max = Core_size[i];
            Core_max = MAX(Core_max, Core_size[i]); 
		}
        // loss type either L1 or LF 
        if( strcmp(argv[5+order],"LF")==0 ){ 
            loss_type = 'F'; 
        } 
        
        // other options for testing 
        int cind = 6+order; 
        double in_lambda = LAMBDA; 
        unsigned int SEED = (unsigned)time(NULL);
        while(cind < argc){
            if(argv[cind][0] == '-'){
                switch(argv[cind][1]){
                    case 'v':
                        VERBOSE = true;
                        break;
                    case 'l':
                        LAMBDA = atof(argv[++cind]);
                        in_lambda = LAMBDA;
                        break;  
                    case 'm':
                        AUTO = false;
                        pratio = atof(argv[++cind]);
                        if(pratio <=0 ){ 
                            pratio = 0;
                            keepPruning = false;  
                        }
                        break;
                    case 't':
   		                threadsN  = atoi(argv[++cind]);         // number of threads to use
                        omp_set_num_threads(threadsN);
                        break;                   
                    case 'f':
  		                FIXEDINIT = true;                       // ONLY FOR TESTING WITH FIXED INITIALIZATION 
                        break;                   
                    case 's':
                        SEED = (unsigned)atoi(argv[++cind]);    // FOR FIXED SEED INITIALIZATION TESTING
                        break; 
                    case 'L':
                        FIXED_LAMBDA = true;                    // Do not adjust lambda value by size content
                        break; 
                    case 'M':
                        MARK = false;                           // Do update pruned elements
                        break; 
                    default: 
                        printf("Unknown option -%c\n\n", argv[cind][1]);
                        break;
               }
            }
            cind++; 
        }
        // initialize random seed based on current time  

        srand(SEED);

		//Getting_Input();
        if(Read_Input() == -1) { printf("nothing can be done; Exiting... \n"); return 0; }

		Vest();

		Print();

        printf("\n[InputINFO]\tlambda\tPrunedRatio\tRE\tTestRE\tTotalTime\tavgIterTime\tNumIter\tFM0ratio\tCore0ratio\tTotal0ratio\tLAMBDA\n");
        printf("[%s,L%c,%d]\t", InputPath,loss_type, AUTO);
        printf("%.3f\t%.3f\t%.3f\t%.5f\t", in_lambda, totalsparsity, RE, TestRE); 
        printf("%.3f\t%.3f\t%d\t", tTime, avgITime, iterNum);  
        printf("%.3f\t%.3f\t%.3f\t", fZR, cZR, tZR);
        printf("%.3f", LAMBDA);         // extra printouts 
	    printf("\n");
        
    }
    // wrong number of arguments 
	else printf("ERROR: Invalid Arguments\n\nUsage: ./VEST [training_input_tensor_path] [test_input_tensor_path] [result_directory_path] [tensor_order(N)] [tensor_rank_1] ... [tensor_rank_N] [truncation_rate] [number of threads]\ne.g.) ./VEST input_train.txt input_test.txt result/ 3 10 10 10 0.5 20\n");
	
    return 0;
}
