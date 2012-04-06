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

#include <lambda/convert.h>


namespace lambda
{

//////////////////
// Constructors //
//////////////////

// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_block_matrix<IndexType,ValueType,MemorySpace>
    ::csr_block_matrix(const MatrixType& matrix)
    {

        lambda::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    csr_block_matrix<IndexType,ValueType,MemorySpace>&
    csr_block_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        lambda::convert(matrix, *this);

        return *this;
    }

} // end namespace lambda

