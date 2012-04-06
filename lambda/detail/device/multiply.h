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

#include <cusp/format.h>
#include <lambda/detail/device/spmv/jad_block.h>


namespace lambda
{
namespace detail
{
namespace device
{


template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              lambda::jad_block_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
#ifdef CUSP_USE_TEXTURE_MEMORY
    lambda::detail::device::spmv_jad_block_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    lambda::detail::device::spmv_jad_block(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif
}

/////////////////
// Entry Point //
/////////////////
template <typename Matrix,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const Matrix&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C)
{
    lambda::detail::device::multiply(A, B, C,
            typename Matrix::format(),
            typename MatrixOrVector1::format(),
            typename MatrixOrVector2::format());
}

} // end namespace device
} // end namespace detail
} // end namespace lambda

