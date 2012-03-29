/*
 * composite_matrix.cu
 *
 *  Created on: Mar 28, 2012
 *      Author: Filippo Squillace
 */

#pragma once


#include <lambda/format.h>


#include <cusp/detail/config.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <lambda/format.h>
#include <cusp/detail/matrix_base.h>


namespace lambda{


template <typename IndexType, typename ValueType, class MemorySpace>
class composite_matrix :
	public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::composite_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::composite_format> Parent;
  public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef lambda::composite_matrix<IndexType, ValueType, MemorySpace2> type; };

    /*! type of the matrices inside the composite matrix
     */
	typedef typename cusp::detail::matrix_base<IndexType, ValueType, MemorySpace, cusp::known_format> Matrix;


    /*
     * Matrices data structures
     */

    Matrix M11;
    Matrix M12;
    Matrix L11;
    Matrix L21;
    Matrix L22;

    /*! Construct an empty \p csr_matrix.
     */
    composite_matrix() {}

    /*! Construct a \p csr_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    composite_matrix(size_t num_rows, size_t num_cols, size_t num_entries)
      : Parent(num_rows, num_cols, num_entries),
        M11(num_rows, num_cols), M12(num_rows, num_cols),\
        L11(num_rows, num_cols), L21(num_rows, num_cols), L22(num_rows, num_cols){}


    template <typename Vector>
    void operator() (Vector& x, Vector& y){
    	printf("ciaoooo\n");
//    	cusp::multiply(M11, x, y);
    }


    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries)
    {
      Parent::resize(num_rows, num_cols, num_entries);
      M11.resize(num_rows, num_cols);
      M12.resize(num_rows, num_cols);
      L11.resize(num_rows, num_cols);
      L21.resize(num_rows, num_cols);
      L22.resize(num_rows, num_cols);

    }

    /*! Swap the contents of two \p csr_matrix objects.
     *
     *  \param matrix Another \p csr_matrix with the same IndexType and ValueType.
     */
    void swap(composite_matrix& matrix)
    {
      Parent::swap(matrix);
      M11.swap(matrix.M11);
      M12.swap(matrix.M12);
      L11.swap(matrix.L11);
      L21.swap(matrix.L21);
      L22.swap(matrix.L22);
    }


}; // class csr_matrix


}
