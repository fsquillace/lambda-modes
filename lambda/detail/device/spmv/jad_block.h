/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>
#include <cusp/detail/device/spmv/jad.h>


#include <thrust/device_ptr.h>

namespace lambda
{
namespace detail
{
namespace device
{

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with
//   the x vector, in parallel.  This division of work implies that
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of
//   work.  Since an entire 32-thread warp is assigned to each row, many
//   threads will remain idle when their row contains a small number
//   of elements.  This code relies on implicit synchronization among
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is
//   used for accessing the x vector.
//
//  Note: THREADS_PER_VECTOR must be one of [2,4,8,16,32]


template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR,
bool UseCache>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR, 1)
__global__ void
spmv_jad_block_kernel(
                       const IndexType num_rows,
                       const IndexType num_cols,
                       const IndexType nb_row,
                       const IndexType nb_col,
                       const IndexType num_blocks,
                       const IndexType block_size,
                       const IndexType num_jagged_diagonals_shape,
                       const IndexType * __restrict__ Ad_shape,
                       const IndexType * __restrict__ Ap_block,
                       const IndexType * __restrict__ Aj_block,
                       const IndexType * __restrict__ Aj,
                       const ValueType * __restrict__ Ax,
                       const ValueType * __restrict__ x,
                             ValueType * __restrict__ y)
{

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;


    __shared__ volatile ValueType sdata[THREADS_PER_BLOCK + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    extern __shared__ IndexType array[];
    volatile IndexType* dia_ptrs = (IndexType*)array;
    volatile IndexType* ptrs_block = (IndexType*)&array[num_jagged_diagonals_shape+1];

    // Get thread info
    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
//    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors


    if (threadIdx.x <= num_jagged_diagonals_shape)
        dia_ptrs[threadIdx.x] = Ad_shape[threadIdx.x];

//CUPRINTF("thread_id=%d thread_lane=%d vec_id=%d vec_lane=%d num_vec=%d nnz_per_block:%d\n",thread_id, thread_lane,
//         vector_id, vector_lane, num_vectors, nnz_per_block);

    if (threadIdx.x <= nb_row)
        ptrs_block[threadIdx.x] = Ap_block[threadIdx.x];

    __syncthreads();


    const IndexType nnz_per_block = dia_ptrs[num_jagged_diagonals_shape];



    for(IndexType row = vector_id; row < num_rows; row += num_vectors){



        const IndexType row_block = row/block_size;
        const IndexType row_local = row%block_size;

        const IndexType start_row_block = ptrs_block[row_block];
        const IndexType lenght_row_block = ptrs_block[row_block+1] - ptrs_block[row_block];

        const IndexType num_elem_blocks = nnz_per_block*start_row_block;
        const IndexType num_elem = row_local*lenght_row_block;

        // initialize local sum
        ValueType sum = 0;
        for(IndexType col_block = thread_lane; col_block<lenght_row_block; col_block+=THREADS_PER_VECTOR){

            IndexType dia_start = dia_ptrs[0];
            IndexType dia_stop = dia_ptrs[1];
            IndexType dia_lenght = dia_stop - dia_start;
            register IndexType dia = 0;
            IndexType dia_off = dia_start*lenght_row_block;

            while((dia_lenght >row_local) && dia < num_jagged_diagonals_shape){
//                if(num_elem_blocks + num_elem + dia_off + col_block>=68564)
//                CUPRINTF("d:%d n_e_b:%d n_e:%d  nnz_per_block:%d tot:%d\n",
//                         dia, num_elem_blocks, num_elem, nnz_per_block,
//
//                         num_elem_blocks + num_elem + dia_off + col_block);
                const IndexType pos = num_elem_blocks + num_elem + dia_off + col_block;

                sum += Ax[pos]*
                    fetch_x<UseCache>(Aj[pos], x);


                dia++;
                if(dia<num_jagged_diagonals_shape){
                    dia_start = dia_stop;
                    dia_stop = dia_ptrs[dia+1];
                    dia_lenght = dia_stop - dia_start;
                    dia_off = dia_start*lenght_row_block;
                }
            }


        }
        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = sdata[threadIdx.x];



    }
}


template <bool UseCache, unsigned int THREADS_PER_VECTOR, typename Matrix, typename ValueType>
void __spmv_jad_block(const Matrix&    A,
                       const ValueType* x,
                             ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    const size_t THREADS_PER_BLOCK  = 256;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
//    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = cusp::detail::device::DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK); //std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    cusp::array1d<ValueType, cusp::device_memory> y_new(A.num_rows);

    if (UseCache)
        bind_x(x);

//printf("THREADS_PER_VECTOR=%d THREADS_PER_BLOCK=%d NUM_BLOCKS=%d\n",
//         THREADS_PER_VECTOR, THREADS_PER_BLOCK, NUM_BLOCKS);

    spmv_jad_block_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache>
    <<<NUM_BLOCKS, THREADS_PER_BLOCK, ((A.num_jagged_diagonals_shape+1)+A.nb_row+1)*sizeof(IndexType)>>>
        (A.num_rows,
         A.num_cols,
         A.nb_row,
         A.nb_col,
         A.num_blocks,
         A.block_size,
         A.num_jagged_diagonals_shape,
         thrust::raw_pointer_cast(&A.diagonal_offsets_shape[0]),
         thrust::raw_pointer_cast(&A.row_offsets_block[0]),
         thrust::raw_pointer_cast(&A.column_indices_block[0]),
         thrust::raw_pointer_cast(&A.column_indices[0]),
         thrust::raw_pointer_cast(&A.values[0]),
         x,
         thrust::raw_pointer_cast(&y_new[0]));



    cusp::detail::device::spmv_jad_permute_kernel<IndexType, ValueType><<<NUM_BLOCKS,THREADS_PER_BLOCK>>>
        (A.num_rows,
         A.num_cols,
         A.num_jagged_diagonals_shape,
         thrust::raw_pointer_cast(&A.permutations[0]),
         thrust::raw_pointer_cast(&y_new[0]),
         y);




    if (UseCache)
        unbind_x(x);
}

template <typename Matrix,
          typename ValueType>
void spmv_jad_block(const Matrix&    A,
                     const ValueType* x,
                           ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    // # of not null blocks per row
    // So, we determine the # of threads in a warp assigned to this blocks.
    const IndexType nnz_blocks_per_row = A.row_offsets_block[A.nb_row] / A.nb_row;

    if (nnz_blocks_per_row <=  2) { __spmv_jad_block<false, 2>(A, x, y); return; }
    if (nnz_blocks_per_row <=  4) { __spmv_jad_block<false, 4>(A, x, y); return; }
    if (nnz_blocks_per_row <=  8) { __spmv_jad_block<false, 8>(A, x, y); return; }
    if (nnz_blocks_per_row <= 16) { __spmv_jad_block<false,16>(A, x, y); return; }

    __spmv_jad_block<false,32>(A, x, y);
}

template <typename Matrix,
          typename ValueType>
void spmv_jad_block_tex(const Matrix&    A,
                         const ValueType* x,
                               ValueType* y)
{
    typedef typename Matrix::index_type IndexType;

    // # of not null blocks per row
    // So, we determine the # of threads in a warp assigned to this blocks.
    const IndexType nnz_blocks_per_row = A.row_offsets_block[A.nb_row] / A.nb_row;

    if (nnz_blocks_per_row <=  2) { __spmv_jad_block<false, 2>(A, x, y); return; }
    if (nnz_blocks_per_row <=  4) { __spmv_jad_block<false, 4>(A, x, y); return; }
    if (nnz_blocks_per_row <=  8) { __spmv_jad_block<false, 8>(A, x, y); return; }
    if (nnz_blocks_per_row <= 16) { __spmv_jad_block<false,16>(A, x, y); return; }

    __spmv_jad_block<true,32>(A, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace lambda

