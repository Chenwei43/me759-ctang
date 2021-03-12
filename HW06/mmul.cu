#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {

	int lda = n, ldb = n, ldc = n;
	const float alphaValue = 1.0, betaValue = 1.0;
	const float* alpha = &alphaValue;
	const float* beta = &betaValue;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc);

}