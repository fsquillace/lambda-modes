/*
 * composite_matrix.h
 *
 *  Created on: Mar 28, 2012
 *      Author: Filippo Squillace
 */

#pragma once


#include <lambda/convert.h>
#include <lambda/format.h>


#include <cusp/detail/matrix_base.h>
#include <cusp/multiply.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

namespace lambda{


template <typename IndexType, typename ValueType, class MemorySpace, typename BaseMatrixType>
class composite_matrix :
	public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::composite_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::composite_format> Parent;
  public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef lambda::composite_matrix<IndexType, ValueType, MemorySpace2, BaseMatrixType> type; };

    /*
     * Matrices data structures
     */

	BaseMatrixType M11;
	BaseMatrixType M12;
	BaseMatrixType L11;
	BaseMatrixType L21;
	BaseMatrixType L22;

    /*! Construct an empty \p composite_matrix.
     */
    composite_matrix() {}

    /*! Construct a \p composite_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    composite_matrix(BaseMatrixType M11, BaseMatrixType M12,\
    		BaseMatrixType L11,	BaseMatrixType L21, BaseMatrixType L22)
      : Parent(M11.num_rows, M11.num_cols,\
    		  M11.num_entries+M12.num_entries+L11.num_entries+L21.num_entries+L22.num_entries){


    	this->M11 = M11;
    	this->M12 = M12;
    	this->L11 = L11;
    	this->L21 = L21;
    	this->L22 = L22;


    }


    /*! Construct a \p composite_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    composite_matrix(size_t num_rows, size_t num_cols, size_t num_entries)
      : Parent(num_rows, num_cols, num_entries),
        M11(num_rows, num_cols), M12(num_rows, num_cols),\
        L11(num_rows, num_cols), L21(num_rows, num_cols), L22(num_rows, num_cols){}


    /*! Construct a \p composite_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    composite_matrix(const MatrixType& matrix);

    template <typename MatrixOrVector1, typename MatrixOrVector2>
    void multiply(MatrixOrVector1& x, MatrixOrVector2& y, cusp::array1d_format, cusp::array1d_format){
    	MatrixOrVector1 omg1(this->num_rows), omg2(this->num_rows),\
    			omg3(this->num_cols), omg4(this->num_rows);

    	y.resize(this->num_cols);

    	// omg1 = M11*x
    	cusp::multiply(M11, x, omg1);

    	// omg2 = L21*x
    	cusp::multiply(L21, x, omg2);

    	// L22*omg3 = omg2
        cusp::default_monitor<ValueType> monitor1(omg2, 100, 1e-6);
        cusp::krylov::cg(L22, omg3, omg2, monitor1);

    	// omg4 = omg1 + M12*omg3
        cusp::multiply(M12, omg3, omg4);
        cusp::blas::axpy(omg1, omg4, ValueType(1));

        // L11*y = omg4
        cusp::default_monitor<ValueType> monitor2(omg2, 100, 1e-6);
        cusp::krylov::cg(L11, y, omg4, monitor2);
    }

    template <typename MatrixOrVector1, typename MatrixOrVector2>
    void operator() (MatrixOrVector1& x, MatrixOrVector2& y){
    	multiply(x, y, typename MatrixOrVector1::format(), typename MatrixOrVector2::format());
    }

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    composite_matrix& operator=(const MatrixType& matrix);


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

    /*! Swap the contents of two \p composite_matrix objects.
     *
     *  \param matrix Another \p composite_matrix with the same IndexType and ValueType.
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


}; // class composite_matrix

}

#include <lambda/detail/composite_matrix.inl>
