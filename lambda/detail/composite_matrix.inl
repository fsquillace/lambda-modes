/*
 * composite_matrix.inl
 *
 *  Created on: Mar 28, 2012
 *      Author: Filippo Squillace
 */


#include <lambda/convert.h>

namespace lambda
{

//////////////////
// Constructors //
//////////////////
        
// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace, typename BaseMatrixType>
template <typename MatrixType>
composite_matrix<IndexType,ValueType,MemorySpace,BaseMatrixType>
    ::composite_matrix(const MatrixType& matrix)
    {
        lambda::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace, typename BaseMatrixType>
template <typename MatrixType>
    composite_matrix<IndexType,ValueType,MemorySpace,BaseMatrixType>&
    composite_matrix<IndexType,ValueType,MemorySpace,BaseMatrixType>
    ::operator=(const MatrixType& matrix)
    {
        lambda::convert(matrix, *this);
        
        return *this;
    }

} // end namespace lambda

