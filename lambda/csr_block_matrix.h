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

/*! \file csr_block_matrix.h
 *  \brief Hybrid Block matrix format
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>

#include <lambda/format.h>

namespace lambda
{
    // Forward definitions
    template <typename IndexType, typename ValueType, class MemorySpace> class csr_block_matrix;

//    template <typename Matrix, typename IndexType, typename ValueType, class MemorySpace> class csr_block_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p csr_matrix : Compressed Sparse Row matrix container
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The matrix entries within the same row must be sorted by column index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p csr_matrix on the host with 6 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 4 nonzeros
 *  cusp::csr_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *  // initialize matrix entries on host
 *  A.row_offsets[0] = 0;  // first offset is always zero
 *  A.row_offsets[1] = 2;
 *  A.row_offsets[2] = 2;
 *  A.row_offsets[3] = 3;
 *  A.row_offsets[4] = 6; // last offset is always num_entries
 *
 *  A.column_indices[0] = 0; A.values[0] = 10;
 *  A.column_indices[1] = 2; A.values[1] = 20;
 *  A.column_indices[2] = 2; A.values[2] = 30;
 *  A.column_indices[3] = 0; A.values[3] = 40;
 *  A.column_indices[4] = 1; A.values[4] = 50;
 *  A.column_indices[5] = 2; A.values[5] = 60;
 *
 *  // A now represents the following matrix
 *  //    [10  0 20]
 *  //    [ 0  0  0]
 *  //    [ 0  0 30]
 *  //    [40 50 60]
 *
 *  // copy to the device
 *  cusp::csr_matrix<int,float,cusp::device_memory> A = B;
 *  \endcode
 *
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class csr_block_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::csr_block_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::csr_block_format> Parent;
    public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef lambda::csr_block_matrix<IndexType, ValueType, MemorySpace2> type; };

    /*! equivalent container type
     */
    typedef typename lambda::csr_block_matrix<IndexType, ValueType, MemorySpace> container;

    /*! type of row offsets indices array
     */
    typedef typename cusp::array1d<IndexType, MemorySpace> row_offsets_array_type;

    /*! type of column indices array
     */
    typedef typename cusp::array1d<IndexType, MemorySpace> column_indices_array_type;

    /*! type of values array
     */
    typedef typename cusp::array1d<ValueType, MemorySpace> values_array_type;

    /*! type of exists_pattern array
     */
    typedef typename cusp::array1d<IndexType, MemorySpace> replicated_block_array_type;

    /*! equivalent view type
     */
//    typedef typename cusp::csr_block_matrix_view<typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>::view,
//                                           IndexType, ValueType, MemorySpace> view;

    /*! equivalent const_view type
     */
//    typedef typename cusp::csr_block_matrix_view<typename cusp::csr_matrix<IndexType,ValueType,MemorySpace>::const_view,
//                                           IndexType, ValueType, MemorySpace> const_view;

    /*! Storage for the row offsets of the CSR data structure.  Also called the "row pointer" array.
     */
    row_offsets_array_type row_offsets_shape;

    /*! Storage for the column indices of the CSR data structure.
     */
    column_indices_array_type column_indices_shape;

    /*! Storage for the nonzero entries of the CSR data structure.
     */
    values_array_type values;


    row_offsets_array_type row_offsets_block;
    column_indices_array_type column_indices_block;


    /*! Number of blocks.
     */
     size_t num_blocks;

    /*! Number of blocks on the column and row.
     */
     size_t nb_col;
     size_t nb_row;

    /*! Size of the blocks on the column and row. The last blocks can have a smaller block size than
        the value of this variable.
     */
     IndexType block_size;

    /*! Construct an empty \p csr_block_matrix.
     */
    csr_block_matrix() {}

    /*! Construct a \p csr_block_matrix with a specific shape and separation into ELL and COO portions.
     *
     *  \param block_siz Size of the block
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    csr_block_matrix(size_t num_rows, size_t num_cols, size_t num_entries,
                     size_t block_size = 2000)
    : Parent(num_rows, num_cols, num_entries)
    {
        // num_rows and num_cols have to be divisible to block_size
        assert(num_rows%block_size==0);
        assert(num_cols%block_size==0);

        // Determine the num of blocks
        nb_col = (num_cols/block_size);
        nb_col += (num_cols%block_size==0)?0:1;

        nb_row = (num_rows/block_size);
        nb_row += (num_rows%block_size==0)?0:1;

        num_blocks = nb_col * nb_row;


        this->block_size = block_size;


        row_offsets_shape(block_size+1);
        column_indices_shape(block_size*block_size);
        values(num_entries);


        row_offsets_block(nb_row+1);
        column_indices_block(num_blocks);

    }

    /*! Construct a \p csr_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    csr_block_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries,
                     size_t block_size = 2000)
    {
        Parent::resize(num_rows, num_cols, num_entries);
        // num_rows and num_cols have to be divisible to block_size
        assert(num_rows%block_size==0);
        assert(num_cols%block_size==0);

        // Determine the num of blocks
        nb_col = (num_cols/block_size);
        nb_col += (num_cols%block_size==0)?0:1;

        nb_row = (num_rows/block_size);
        nb_row += (num_rows%block_size==0)?0:1;

        num_blocks = nb_col * nb_row;


        this->block_size = block_size;


        row_offsets_shape.resize(block_size+1);
        column_indices_shape.resize(block_size*block_size);
        values.resize(num_entries);


        row_offsets_block.resize(nb_row+1);
        column_indices_block.resize(num_blocks);

    }

    /*! Swap the contents of two \p csr_block_matrix objects.
     *
     *  \param matrix Another \p csr_block_matrix with the same IndexType and ValueType.
     */
    void swap(csr_block_matrix& matrix)
    {
        Parent::swap(matrix);
        row_offsets_shape.swap(matrix.row_offsets_shape);

        column_indices_shape.swap(matrix.column_indices_shape);

        values.swap(matrix.values);

        row_offsets_block.swap(matrix.row_offsets_block);
        column_indices_block.swap(matrix.column_indices_block);
    }

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    csr_block_matrix& operator=(const MatrixType& matrix);
}; // class csr_matrix
/*! \}
 */

// WARNING: Could be useless
///*! \addtogroup sparse_matrix_views Sparse Matrix Views
// *  \ingroup sparse_matrices
// *  \{
// */
//
///*! \p csr_bock_matrix_view : Hybrid Block ELL/COO matrix view
// *
// * \tparam Matrix Type of \c hyb
// * \tparam IndexType Type used for matrix indices (e.g. \c int).
// * \tparam ValueType Type used for matrix values (e.g. \c float).
// * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
// *
// */
//template <typename Matrix,
//          typename IndexType   = typename Matrix::index_type,
//          typename ValueType   = typename Matrix::value_type,
//          typename MemorySpace = typename cusp::minimum_space<typename Matrix::memory_space, typename Matrix::memory_space>::type >
//class csr_block_matrix_view : public detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>
//{
//  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format> Parent;
//  public:
//    /*! type of \p HYB portion of the HYB structure
//     */
//    typedef Matrix csr_matrix_type;
//
//    /*! equivalent container type
//     */
//    typedef typename cusp::csr_matrix<IndexType, ValueType, MemorySpace> container;
//
//    /*! equivalent view type
//     */
//    typedef typename cusp::csr_matrix_view<Matrix, IndexType, ValueType, MemorySpace> view;
//
//    /*! View to the \p HYB portion of the HYB structure.
//     */
//    csr_matrix_type* hyb;
//
//    /*! Construct an empty \p csr_block_matrix_view.
//     */
//    csr_block_matrix_view() {}
//
//    template <typename OtherMatrix>
//    csr_block_matrix_view(OtherMatrix& hyb)
//    : Parent(hyb.num_rows, hyb.num_cols, hyb.num_entries), ell(ell), coo(coo) {}
//
//    template <typename OtherMatrix>
//    csr_block_matrix_view(const OtherMatrix& hyb)
//    : Parent(ell.num_rows, ell.num_cols, ell.num_entries + coo.num_entries), ell(ell), coo(coo) {}
//
//    template <typename Matrix>
//    csr_block_matrix_view(Matrix& A)
//    : Parent(A), ell(A.ell), coo(A.coo) {}
//
//    template <typename Matrix>
//    csr_block_matrix_view(const Matrix& A)
//    : Parent(A), ell(A.ell), coo(A.coo) {}
//
//    /*! Resize matrix dimensions and underlying storage
//     */
//    void resize(size_t num_rows, size_t num_cols,
//                size_t num_ell_entries, size_t num_coo_entries,
//                size_t num_entries_per_row, size_t alignment = 32)
//    {
//      Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
//      ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
//      coo.resize(num_rows, num_cols, num_coo_entries);
//    }
//};
///*! \} // end Views
// */

} // end namespace lambda

#include <lambda/detail/csr_block_matrix.inl>

