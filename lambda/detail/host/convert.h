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
#include <cusp/csr_matrix.h>

#include <lambda/detail/host/conversion.h>
#include <lambda/format.h>
#include <lambda/convert.h>

#include <algorithm>
#include <string>
#include <stdexcept>


#include <stdio.h>
#include <stdlib.h>


namespace lambda
{
namespace detail
{
namespace host
{

// Host Conversion Functions
// CSR <- JAD_BLOCK

///////////////
// JAD BLOCK //
///////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             lambda::jad_block_format,
             size_t block_size = 2000)
{
    char * bs;
    bs = getenv ("BLOCK_SIZE");
    if (bs==NULL)
        block_size=2000;
    else
        block_size = atoi(bs);
//    block_size=5203; //2047; //5203;
    size_t minim = thrust::min<size_t>(src.num_cols, src.num_rows);
    block_size = thrust::min<size_t>(block_size, minim);
    lambda::detail::host::csr_to_jad_block(src, dst, block_size);
}

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             lambda::jad_block_format)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::convert(src, csr);
    lambda::convert(csr, dst);
}


/////////////////
// Entry Point //
/////////////////

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    lambda::detail::host::convert(src, dst,
            typename Matrix1::format(),
            typename Matrix2::format());
}

} // end namespace host
} // end namespace detail
} // end namespace lambda

