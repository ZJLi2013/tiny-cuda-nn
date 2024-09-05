#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>

#include <cublas_v2.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream> 
#include <type_traits>

#include <tiny-cuda-nn/debug_config.h> // for cublas matrix print 

namespace tcnn {

#define Cublas_CHECK_THROW(status) \
    do { \
        cublasStatus_t _status = (status); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(_status)); \
        } \
    } while (0)

using TypeAccumulator = std::conditional_t<std::is_same<network_precision_t, float>::value, float, __half>;
using TypeCompute = std::conditional_t<std::is_same<network_precision_t, float>::value, float, __half>;

template <typename V, int Count>
struct CublasFragmentWrapper {
	static const uint32_t num_elements = Count;
	V x[Count];
};

template<typename ElementAccumulator, typename MyFragment, int kCount>
__global__ void create_fragments(ElementAccumulator* matrixC,  MyFragment* d_fragments, int m, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    int offset = idx * kCount ;
    if(offset < m *n){
        TCNN_PRAGMA_UNROLL
        for(int i=0; i<kCount ; ++i){
            int element_idx = offset + i ;
            if(element_idx < m*n){
                d_fragments[idx].x[i] = matrixC[element_idx]; 
            }
        }
    }
}

template<typename ElementAccumulator, typename MyFragment, int kCount>
__global__ void merge_fragments(MyFragment* d_fragments, ElementAccumulator* matrixC,  int m, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    int offset = idx * kCount ;
    if(offset < m *n ){
        TCNN_PRAGMA_UNROLL
        for(int i=0; i<kCount; ++i){
            int element_idx = offset + i ;
            if( element_idx < m*n){
                matrixC[element_idx] = d_fragments[idx].x[i]; 
            }
        }
    }       
}

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_          ///< Data type used to compute linear combination
>
class ActivationEpilogue {
public: 
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;
    static int const kCount = Count ; 
	struct Params {
		Activation activation;
		bool sum_source;
	};
public:
	ActivationEpilogue(Params const &params) : m_activation{params.activation}, m_sum_source{params.sum_source} { }
	bool is_source_needed() const {
		return m_sum_source;
	}
    using MyFragment = CublasFragmentWrapper<ElementAccumulator, kCount> ;
    ElementOutput operator()(ElementAccumulator* accumulator, int m, int n) const {
            int threads_per_block = 256 ;
            int total_threads = ( m *n + kCount - 1) / kCount; 
            int blocks = (total_threads + thrads_per_block -1) / threads_per_block ;
            // Temp Solution 
            myFrag* d_fragments ; 
            int num_fragments =  ( m *n + kCount - 1) / kCount; 
            cudaMalloc(&d_fragments, num_fragments * sizeof(myFrag));
            create_fragments<ElementAccumulator, MyFragment, kCount>(accumulator, d_fragments, m, n); 
            activation_kernel<T, myFrag, kCount><<<blocks, threads_per_block>>>(m_activation, d_fragments, m, n); 
            cudaStreamSynchrnize(); 
            merge_fragments<ElementAccumulator, MyFragment, kCount>(d_fragments, accumulator, m, n);
            cudaFree(d_fragments);
    } 

    ElementOutput operator()(ElementAccumulator* accumulator, ElementOutput* source,  int m, int n) const {
        std::cout << "NOT Implement in ActivationEpilogue" << std::endl ;
    } 

private:
	Activation m_activation;
	bool m_sum_source;

}; 

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_          ///< Data type used to compute linear combination
>
class ActivationTransferEpilogue {
public: 
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;

    static int const kCount = Count ; 

    using FragmentOutput = std::array<ElementOutput_, kCount>; 
	using FragmentAccumulator = std::array<ElementAccumulator, kCount>;
	using ComputeFragment = std::array<ElementCompute, kCount>;   

	struct Params {
		Activation activation;
	};
public:
	ActivationTransferEpilogue(Params const &params) : m_activation{params.activation} { }

	bool is_source_needed() const {
		return true;
	}

    using MyFragment = CublasFragmentWrapper<ElementAccumulator, kCount> ;
    /*
        accumulator :: dL/da at each layer during backward pass  
        source :: input for activation_Op at each layer during forward pass 
        TODO: what's and how accumulator, source passed into this epilogue after cublas::gemm done ?
    */
    ElementOutput operator()(ElementAccumulator* accumulator, ElementOutput* source, int m, int n) const {
            int threads_per_block = 256 ;
            int total_threads = ( m *n + kCount - 1) / kCount; 
            int blocks = (total_threads + thrads_per_block -1) / threads_per_block ;
            // Temp Solution 
            myFrag* a_fragments ; 
            myFrag* s_fragments; 
            int num_fragments =  ( m *n + kCount - 1) / kCount; 
            CUDA_CHECK_THROW(cudaMalloc(&a_fragments, num_fragments * sizeof(myFrag)));
            CUDA_CHECK_THROW(cudaMalloc(&s_fragments, num_fragments * sizeof(myFrag))); 
            create_fragments<ElementAccumulator, MyFragment, kCount>(accumulator, a_fragments, m, n); 
            create_fragments<ElementOutput, MyFragment, kCount>(source, s_fragments, m, n); 
            activation_backward_kernel<T, myFrag, kCount><<<blocks, threads_per_block>>>(m_activation, a_fragments, m, n); 
            cudaStreamSynchrnize(); 
            merge_fragments<ElementAccumulator, MyFragment, kCount>(a_fragments, accumulator, m, n);
            // s_fragments is used as const only for activaton_backward_kernel, so direct delete once done 
            CUDA_CHECK_THROW(cudaFree(d_fragments));
            CUDA_CHECK_THROW(cudaFree(s_fragments));
    }    

    ElementOutput operator()(ElementAccumulator* accumulator, int m , int n ){
        std::cout << "NOT Implement 2" << std::endl; 
    }

private:
	Activation m_activation;
	bool m_sum_source;
}; 

template <typename T>
// if using tensorCore, vec as 32, if using cuda cores, vec as 1 
static constexpr int n_vectorized_elements = (! std::is_same<T, float>::value) ? (128 / sizeof(T)) : 1;

template <typename T>
using ActivationOp = ActivationEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationTransferOp = ActivationTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

// template structure ofr gemm op
template<typename EPILOGUE, typename T>
struct OurGemmWrapper; 

cudaDataType_t getCUDADatatype(const std::type_info &type)
{
    if (type == typeid(float)){
        return CUDA_R_32F ;
    }else if (type == typeid(__half)){
        return CUDA_R_16F; 
    }
    std::cout << "Unsupported data type" << std::endl;
    exit(EXIT_FAILURE);
}

// minic cutlass::epilogue
// as cutlass::epilogue op is smoothly using previous GEMM grid/block/warps mapping.
// basically the fragment memory operated by the warp and threads in the warp can directly used for epilogue op again.

template<typename EPILOGUE, typename T>
void OurGemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc) {
    
    cudaDataType_t dataType = CUDA_R_32F;
    if (! std::is_same<T, float>::value){
        cudaDataType_t dataType = getCUDADatatype(typeid(__half)); 
    } 
    cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                         m, n, k,
                                         alpha,
                                         A, dataType, lda,
                                         B, dataType, ldb,
                                         beta,
                                         C, dataType, ldc,
                                         dataType, // Compute type
                                         CUBLAS_GEMM_DEFAULT);
                                         
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed");
    }
    
#ifdef DEBUG_MODE    
    std::cout << "gemm output before epilogue" << std::endl ; 
    printMatrix(C, m, n); 
#endif  

    // do activation op on matrix C 
    EPILOGUE myActivation ; 
    if ( myActivation.is_source_needed()){
        //TODO: need double check, should the source matrix be C ?
        myActivation(C, C, m, n); 
    }else{
        myActivation(C, m, n); 
    }
    std::cout << "[DEBUG]: done activation after gemm op" << std::endl; 

#ifdef DEBUG_MODE   
    std:cout << "final gemm output after epilogue" << std::endl ;
    printMatrix(C, m, n);
#endif 

}

// specialization for float OurGemm 
template<typename EPILOGUE>
struct OurGemmWrapper<EPILOGUE, float>{
    static cublasStatus_t gemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc)
    {
        return OurGemm<EPILOGUE, float>(handle, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }
}; 

// specialization for __half OurGemm 
template<typename EPILOGUE>
struct OurGemmWrapper<EPILOGUE, __half>{
    static cublasStatus_t gemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc)
    {
        return OurGemm<EPILOGUE, __half>(handle, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }
}; 

template<typename T>
void OurSplitGemm(cublasHandle_t handle,      
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  const int m, const int n, const int k,
                  const void* alpha,
                  const T *A, int lda,
                  const T *B, int ldb,
                  const void* beta,
                  T *C, int ldc,
                  int split_k_slices)
{
    cudaDataType_t dataType = getCUDADatatype(typeid(network_precision_t));
    if (split_k_slices == 1){
        // std::cout << "[DEBUG: split_k_slices=1 for debug]" << std::endl ;
        cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                            m, n, k,
                                            alpha,
                                            A, dataType, lda,
                                            B, dataType, ldb,
                                            beta,
                                            C, dataType, ldc,
                                            dataType, 
                                            CUBLAS_GEMM_DEFAULT);
                                            
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS GEMM failed");
        }        
    }else{
        // Split-K GEMM implementation
        const T* B_slice;
        const T* A_slice;   
        int k_slice = k / split_k_slices;
        for (int slice = 0; slice < split_k_slices; ++slice) {
            //TODO: data access consider layout 
            A_slice = static_cast<const T*>(A) + slice * k_slice ;
            B_slice = static_cast<const T*>(B) + slice * k_slice * ldb;
            cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                                m, n, k_slice,
                                                alpha,
                                                A_slice, dataType, lda,
                                                B_slice, dataType, ldb,
                                                beta,
                                                C, dataType, ldc,
                                                dataType, 
                                                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS GEMM failed");
            } 
        }
    }
};

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {

     cublasOperation_t TransA = (LayoutA == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
     cublasOperation_t TransB = (LayoutB == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
     cublasOperation_t TransC = (LayoutC == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;

    static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");

	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, __half>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, __half>;    

    if (A.n() != B.m()) {
        throw std::runtime_error("Matrices A and B cannot be multiplied together");
    }

    const int M = A.m();
    const int K = A.n();
    const int N = B.n();

    if (C.m() != M || C.n() != N) {
        throw std::runtime_error(fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N));
    }

    if (D.m() != M || D.n() != N) {
        throw std::runtime_error(fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N));
    }

    // int lda = (LayoutA == RM) ? K : M;
    // int ldb = (LayoutB == RM) ? N : K;
    // int ldc = (LayoutC == RM) ? N : M;
    int lda = M ;
    int ldb = K ;
    int ldc = M ; 

    network_precision_t alpha = 1.0f;
    network_precision_t beta = sum_source ? 1.0f : 0.0f;

    if(transfer){
        OurGemmWrapper<ActivationTransferOp<MatmulTypeAccumulator>, network_precision_t>::gemm(handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
    }else{
        // TODO: sum_source op before epilogue  
        OurGemmWrapper<ActivationOp<MatmulTypeAccumulator>, network_precision_t>::gemm(handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc);
    }
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply(handle, stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
	} else {
		fc_multiply(handle, stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (B.layout() == CM) {
		fc_multiply(handle, stream, A, B.cm(), C, D, act, transfer, sum_source);
	} else {
		fc_multiply(handle, stream, A, B.rm(), C, D, act, transfer, sum_source);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None) {
	fc_multiply(handle, stream, A, B, D, D, act);
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, const GPUMatrix<TypeD, LayoutD>& D, int split_k_slices = 1, float beta = 0.0f) {
    
    cublasOperation_t TransA = (LayoutA == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t TransB = (LayoutB == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t TransC = (LayoutC == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;   
    
    static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
 
 	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, __half>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, __half>;    

    if (A.n() != B.m()) {
        throw std::runtime_error("Matrices A and B cannot be multiplied together");
    }

    const int M = A.m();
    const int K = A.n();
    const int N = B.n();

    if (C.m() != M || C.n() != N) {
        throw std::runtime_error(fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N));
    }

    if (D.m() != M || D.n() != N) {
        throw std::runtime_error(fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N));
    }

    // int lda = (LayoutA == RM) ? K : M;
    // int ldb = (LayoutB == RM) ? N : K;
    // int ldc = (LayoutC == RM) ? N : M;
    // A(m, k), B(k, n), C(m, n) , leadning-dim only relate to physical memory layout, no matter T or N 
    int lda = M ; 
    int ldb = K / split_k_slices;
    int ldc = M ; 
    // cublasSetStream(handle, stream);
    // TODO: need specify ComputeType and AccumulatorType 
    network_precision_t alpha = __float2half(1.0) ; 
    network_precision_t half_beta = __float2half(1.0) ;  // for splitK case, need to accumulate C from each split to form final C matrix 
    OurSplitGemm<network_precision_t>(handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &half_beta, C.data(), ldc, split_k_slices); 
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply_split_k<TypeA, LayoutA, TypeB, LayoutB, TypeC, CM, TypeD, CM>(handle, stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {        
		fc_multiply_split_k<TypeA, LayoutA, TypeB, LayoutB, TypeC, RM, TypeD, RM>(handle, stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(handle, stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(handle, stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(handle, stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(handle, stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename TypeA, typename TypeB, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, int split_k_slices, float beta) {
	fc_multiply_split_k(handle, stream, A, B, D, D, split_k_slices, beta);
}

}