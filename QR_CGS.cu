#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime_api.h>

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::generate;
using std::vector;

/* ********************************************************* */
/* These two lines define the dimensions (MxN) of the matrix */
#define M 4 // Number of elements/vector
#define N 3 // Number of vectors
/* Change them to test different size matrices               */
/* ********************************************************* */

/* CPU Functions */
void matrixTranspose(vector<float>& V, vector<float>& Vt, bool reverse);
void printMatrix(vector<float>& V);
void printTranspose(vector<float>& Vt);

// Function to run CGS and QR decomposition
void runQR();
vector<float> runCGS();

/* GPU Functions */
__global__ void printVectorKernel(float* v);
__global__ void printMatrixKernel(float* V);
__global__ void getVectorKernel(float* v, float* V_t, int rowNum, bool reverse);
__global__ void matrixTransposeKernel(float* V, float* V_t, bool reverse);

__global__ void matrixMultGPU(float* Q_t, float* A, float* R);

__global__ void calculateProjectionGPU(float* u, float* upper, float* lower, float* p);
__global__ void innerProductGPU(float* a, float* b, float* c);
__global__ void sumProjectionsGPU(float* P_t, float* projSum);
__global__ void vectorSubGPU(float* v, float* projSum, float* u);
__global__ void vectorNormsGPU(float* U_t, float* norms);
__global__ void normMultGPU(float* U, float* norms, float* E);

int main() {

	runQR();

	return 0;
}

/* CPU Functions: */
// Transposes or reverse-transposes a matrix passed in as a 1D vector
void matrixTranspose(vector<float>& V, vector<float>& Vt, bool reverse) {

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (!reverse) {
				Vt[i * M + j] = V[j * N + i];
			}
			else {
				V[j * N + i] = Vt[i * M + j];
			}
		}
	}
}

// Prints a matrix passed in as a 1-D vector
void printMatrix(vector<float>& V) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << V[i * N + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

// Prints the transpose of a matrix passed in as a 1-D vector
void printTranspose(vector<float>& Vt) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			cout << Vt[i * M + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

/* GPU Functions: */
// Prints a vector from the GPU
__global__ void printVectorKernel(float* v) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			printf("%f\t", v[i]);
		}
		printf("\n");
	}
}

// Prints a matrix from the GPU
__global__ void printMatrixKernel(float* V) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				printf("%f\t", V[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

// Transposes or reverse-transposes a matrix from GPU
__global__ void matrixTransposeKernel(float* V, float* V_t, bool reverse) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				if (!reverse) {
					V_t[i * M + j] = V[j * N + i];
				}
				else {
					V[j * N + i] = V_t[i * M + j];
				}
			}
		}
	}
}

// Accesses a row in V_transpose and copies it into the storage vector v or does the reverse
__global__ void getVectorKernel(float* v, float* V_t, int rowNum, bool reverse) {

	if (threadIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			if (!reverse) {
				v[i] = V_t[rowNum * M + i];
			}
			else {
				V_t[rowNum * M + i] = v[i];
			}
		}
	}
}

// Multiply a vector by a scalar to get a projection - requires M threads for M-length vectors
__global__ void calculateProjectionGPU(float* u, float* upper, float* lower, float* p) {

	int i = threadIdx.x;
	// Each thread does one multiplication
	if (i < M) {
		if (*lower != 0) {
			__shared__ float temp[M];
			temp[i] = *upper / *lower;
			__syncthreads();
			p[i] = u[i] * temp[i];
		}
		else {
			p[i] = 0.0f;
		}
	}
}

// Calculate inner product on GPU - basically stolen from https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
__global__ void innerProductGPU(float* a, float* b, float* c) {

	// Likely to have more threads than entires, so use this to keep in range
	if (threadIdx.x < M) {
		// Each thread does one multiplication
		// Need to use shared memory to store products
		__shared__ float temp[M];
		temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
		// Need threads to synchronize - no threads advance until all are at this line, ensures no read-before-write hazard
		__syncthreads();
		// Now do the sum using only thread 0
		if (threadIdx.x == 0) {
			float sum = 0.0f;
			for (int i = 0; i < M; i++) {
				sum += temp[i];
			}
			*c = sum;
		}
	}
}

// Adds all of the projections onto one u of each v and returns that vector - requires M threads for M-length vectors
__global__ void sumProjectionsGPU(float* P_t, float* projSum) {

	int idx = threadIdx.x;
	if (idx < M) {
		float temp = 0.0f;
		for (int i = 0; i < N; i++) {
			temp += P_t[i * M + idx];
		}
		projSum[idx] = temp;
	}
}

// Vector subtraction to get u[i] - requires M threads for M-length vectors, will be executed from 1 thread
__global__ void vectorSubGPU(float* v, float* projSum, float* u) {

	int i = threadIdx.x;
	// Each thread subtracts one element from the other
	if (i < M) {
		u[i] = v[i] - projSum[i];
	}
}

// Calculates the eculidean norms of each vector and stores them into array - requires N threads for N columns
__global__ void vectorNormsGPU(float* U_t, float* norms) {

	int idx = threadIdx.x;
	if (idx < N) {
		float temp = 0.0f;
		// First sum the components of each u together
		for (int i = 0; i < M; i++) {
			temp += (U_t[idx * M + i] * U_t[idx * M + i]);
		}
		// Now get reciprocal sqrt and store into norms array
		norms[idx] = rsqrtf(temp);
	}
}

// Mulitiplies each u by 1/norm to get the e's - requires M*N threads to do all at once
__global__ void normMultGPU(float* U, float* norms, float* E) {

	// Note: This function requires that U be passed in, not U_t (for indexing purposes)
	int idx = threadIdx.x;
	if (idx < M * N) {
		// Get index in norms array
		int normIdx = (idx % N);
		E[idx] = U[idx] * norms[normIdx];
	}
}

__global__ void matrixMultGPU(float* Q_t, float* A, float* R) {

	// Get each thread x and y
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	// Q_t is a NxM matrix, A is a MxN matrix
	// Therefore R will be NxN
	if ((row < N) && (col < N)) {
		float sum = 0.0f;
		for (int i = 0; i < M; i++) {
			R[row * N + col] += Q_t[row*M + i] * A[i*N + col];
		}
	}
}

vector<float> runCGS() {

	size_t bytesIP = sizeof(float); // One float
	size_t bytesMatrix = M * N * sizeof(float); // MxN matrix
	size_t bytesVecLen = M * sizeof(float); // Mx1 vector
	size_t bytesNumVec = N * sizeof(float); // Nx1 vector

	// Initialize vectors and matrices for QR
	vector<float> h_v(M, 0.0f); // Storage vector for calculating u's
	vector<float> h_V(M * N); // Input matrix of v's
	vector<float> h_Vt(M * N); // Transpose of V
	vector<float> h_u(M, 0.0f); // Storage vector for calculating u's
	vector<float> h_U(M * N, 0.0f); // Initially empty matrix of u's
	vector<float> h_Ut(M * N, 0.0f); // Transpose of U

	float* h_Upper = nullptr;
	float* h_Lower = nullptr;

	vector<float> h_p(M, 0.0f); // Holds a single projection
	vector<float> h_Pt(M * N, 0.0f); // Transpose of projections matrix
	vector<float> h_PS(M, 0.0f); // Sum of projections vector

	vector<float> h_N(N, 0.0f); // Vector of norms
	vector<float> h_E(M * N, 0.0f); // Output E matrix

	// Initialize V with a 4x3 example that works out nicely - http://www.cs.nthu.edu.tw/~cherung/teaching/2008cs3331/chap4%20example.pdf
	h_V[0] = 1.0; h_V[1] = -1.0; h_V[2] = 4.0;
	h_V[3] = 1.0; h_V[4] = 4.0; h_V[5] = -2.0;
	h_V[6] = 1.0; h_V[7] = 4.0; h_V[8] = 2.0;
	h_V[9] = 1.0; h_V[10] = -1.0; h_V[11] = 0.0;

	// Initialize V_transpose
	matrixTranspose(h_V, h_Vt, false);
	// Copy v1 to u1
	for (int i = 0; i < M; i++) {
		h_Ut[i] = h_Vt[i];
	}
	// Store into h_U
	matrixTranspose(h_U, h_Ut, true);

	// Print initial V matrix:
	printf("Input Matrix V:\n");
	printMatrix(h_V);
	printf("\n");

	// Allocate device memory
	float* d_v, * d_V, * d_Vt, * d_u, * d_U, * d_Ut, * d_Upper, * d_Lower, * d_p, * d_Pt, * d_PS, * d_N, * d_E;
	cudaMalloc(&d_v, bytesVecLen);
	cudaMalloc(&d_V, bytesMatrix);
	cudaMalloc(&d_Vt, bytesMatrix);
	cudaMalloc(&d_u, bytesVecLen);
	cudaMalloc(&d_U, bytesMatrix);
	cudaMalloc(&d_Ut, bytesMatrix);
	cudaMalloc((void**)&d_Upper, bytesIP);
	cudaMalloc((void**)&d_Lower, bytesIP);
	cudaMalloc(&d_p, bytesVecLen);
	cudaMalloc(&d_Pt, bytesMatrix);
	cudaMalloc(&d_PS, bytesVecLen);
	cudaMalloc(&d_N, bytesNumVec);
	cudaMalloc(&d_E, bytesMatrix);

	cudaMemcpy(d_v, h_v.data(), bytesVecLen, cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V.data(), bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vt, h_Vt.data(), bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, h_u.data(), bytesVecLen, cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, h_U.data(), bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ut, h_Ut.data(), bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Upper, h_Upper, bytesIP, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Lower, h_Lower, bytesIP, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, h_p.data(), bytesVecLen, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pt, h_Pt.data(), bytesMatrix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_PS, h_PS.data(), bytesVecLen, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N.data(), bytesNumVec, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, h_E.data(), bytesMatrix, cudaMemcpyHostToDevice);

	int numBlocks = 1;
	int threads = 32;
	dim3 threadsPerBlock(threads, 1, 1);

	// Iterating over each of the N columns:
		// 1. Calculate projections of v(i) onto previous u(i)'s
		// 2. Sum those projections
		// 3. Subtract that sum of projections from v(i) to get next u(i)
		// 4. Repeat 1-3

	for (int i = 1; i < N; i++) {
		// Get next vi and store it into d_v
		getVectorKernel << <numBlocks, threadsPerBlock >> > (d_v, d_Vt, i, false);
		cudaDeviceSynchronize();

		// Calculate projections of vi onto the u's
		for (int j = 0; j < N; j++) {
			// Get column j's u vector
			getVectorKernel << <numBlocks, threadsPerBlock >> > (d_u, d_Ut, j, false);
			cudaDeviceSynchronize();
			// Computer the two inner products for the projection onto that u
			innerProductGPU << <numBlocks, threadsPerBlock >> > (d_u, d_v, d_Upper);
			innerProductGPU << <numBlocks, threadsPerBlock >> > (d_u, d_u, d_Lower);
			cudaDeviceSynchronize();
			// Compute the projection and store into d_p
			calculateProjectionGPU << <numBlocks, threadsPerBlock >> > (d_u, d_Upper, d_Lower, d_p);
			cudaDeviceSynchronize();
			// Store d_p into d_Pt
			getVectorKernel << <numBlocks, threadsPerBlock >> > (d_p, d_Pt, j, true);
			cudaDeviceSynchronize();
		}
		// Next sum projections
		sumProjectionsGPU << <numBlocks, threadsPerBlock >> > (d_Pt, d_PS);
		cudaDeviceSynchronize();
		// Now calculate next u
		vectorSubGPU << <numBlocks, threadsPerBlock >> > (d_v, d_PS, d_u);
		cudaDeviceSynchronize();
		// Now place that u into U_t
		getVectorKernel << <numBlocks, threadsPerBlock >> > (d_u, d_Ut, i, true);
		cudaDeviceSynchronize();
		// Reverse transpose U_t
		matrixTransposeKernel << <numBlocks, threadsPerBlock >> > (d_U, d_Ut, true);
		cudaDeviceSynchronize();
		printf("\n");
	}

	// Next calculate norms
	vectorNormsGPU << <numBlocks, threadsPerBlock >> > (d_Ut, d_N);
	cudaDeviceSynchronize();
	// Finally get output E matrix
	normMultGPU << <numBlocks, threadsPerBlock >> > (d_U, d_N, d_E);
	cudaDeviceSynchronize();

	// Copy output E matrix back
	cudaMemcpy(h_E.data(), d_E, bytesMatrix, cudaMemcpyDeviceToHost);

	// Free memory on device
	cudaFree(d_v);
	cudaFree(d_V);
	cudaFree(d_Vt);
	cudaFree(d_u);
	cudaFree(d_U);
	cudaFree(d_Ut);
	cudaFree(d_Upper);
	cudaFree(d_Lower);
	cudaFree(d_Pt);
	cudaFree(d_PS);
	cudaFree(d_N);
	cudaFree(d_E);

	return h_E;
}

void runQR() {

	vector<float> h_Q(M * N, 0.0f);
	h_Q = runCGS();
	// Print E matrix
	printf("Orthonormal Basis Q: \n");
	printMatrix(h_Q);
	printf("\n");

	// Get transpose of Q
	vector<float> h_Qt(M * N, 0.0f);
	matrixTranspose(h_Q, h_Qt, false);
	printf("Transpose of Q: \n");
	printTranspose(h_Qt);
	printf("\n");

	// Init GPU Parameters
	int numBlocks = 1;
	int threads = 32;
	dim3 threadsPerBlock(threads, threads);

	size_t bytesQt = N * M * sizeof(float);
	size_t bytesA = M * N * sizeof(float);
	size_t bytesR = N * N * sizeof(float);

	vector<float> h_A(M * N, 0.0f);
	h_A[0] = 1.0; h_A[1] = -1.0; h_A[2] = 4.0;
	h_A[3] = 1.0; h_A[4] = 4.0; h_A[5] = -2.0;
	h_A[6] = 1.0; h_A[7] = 4.0; h_A[8] = 2.0;
	h_A[9] = 1.0; h_A[10] = -1.0; h_A[11] = 0.0;
	vector<float> h_R(N * N, 0.0f);
	// Allocate and copy to device memory
	float* d_Qt, * d_A, * d_R;
	cudaMalloc(&d_Qt, bytesQt);
	cudaMalloc(&d_A, bytesA);
	cudaMalloc(&d_R, bytesR);

	cudaMemcpy(d_Qt, h_Qt.data(), bytesQt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R.data(), bytesR, cudaMemcpyHostToDevice);

	// Run the matrix multiplication
	matrixMultGPU << <numBlocks, threadsPerBlock >> > (d_Qt, d_A, d_R);

	// Copy data back
	cudaMemcpy(h_R.data(), d_R, bytesR, cudaMemcpyDeviceToHost);

	// Print R
	printf("Upper triangular matrix R:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f\t", h_R[i * N + j]);
		}
		printf("\n");
	}
	printf("\n");
}