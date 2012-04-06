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

/*! \file jad_block_matrix.h
 *  \brief Hybrid Block matrix format
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/detail/matrix_base.h>

#include <lambda/format.h>
#include <lambda/multiply.h>


namespace lambda
{
    // Forward definitions
    template <typename IndexType, typename ValueType, class MemorySpace> class jad_block_matrix;

//    template <typename Matrix, typename IndexType, typename ValueType, class MemorySpace> class jad_block_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p jad_matrix : Compressed Sparse Row matrix container
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The matrix entries within the same row must be sorted by column index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p jad_matrix on the host with 6 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 *  #include <cusp/jad_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 4 nonzeros
 *  cusp::jad_matrix<int,float,cusp::host_memory> A(4,3,6);
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
 *  cusp::jad_matrix<int,float,cusp::device_memory> A = B;
 *  \endcode
 *
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class jad_block_matrix : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::jad_block_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,lambda::jad_block_format> Parent;
    public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef lambda::jad_block_matrix<IndexType, ValueType, MemorySpace2> type; };

    /*! equivalent container type
     */
    typedef typename lambda::jad_block_matrix<IndexType, ValueType, MemorySpace> container;

    /*! type of row offsets indices array
     */
    typedef typename cusp::array1d<IndexType, MemorySpace> index_array_type;

    /*! type of values array
     */
    typedef typename cusp::array1d<ValueType, MemorySpace> values_array_type;


    /*! equivalent view type
     */
//    typedef typename cusp::jad_block_matrix_view<typename cusp::jad_matrix<IndexType,ValueType,MemorySpace>::view,
//                                           IndexType, ValueType, MemorySpace> view;

    /*! equivalent const_view type
     */
//    typedef typename cusp::jad_block_matrix_view<typename cusp::jad_matrix<IndexType,ValueType,MemorySpace>::const_view,
//                                           IndexType, ValueType, MemorySpace> const_view;

    /*! Storage for the row offsets of the jad data structure.  Also called the "row pointer" array.
     */
    index_array_type diagonal_offsets_shape;

    /*! Storage for the column indices of the jad data structure.
     */
    index_array_type column_indices;

    /*! Storage for the nonzero entries of the jad data structure.
     */
    values_array_type values;

    index_array_type permutations;

    index_array_type row_offsets_block;
    index_array_type column_indices_block;

    size_t num_jagged_diagonals_shape;

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

    /*! Construct an empty \p jad_block_matrix.
     */
    jad_block_matrix() {}

    /*! Construct a \p jad_block_matrix with a specific shape and separation into ELL and COO portions.
     *
     *  \param block_siz Size of the block
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    jad_block_matrix(size_t num_rows, size_t num_cols, size_t num_entries,
                     size_t num_jagged_diagonals_shape, size_t block_size = 2000)
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
        this->num_jagged_diagonals_shape = num_jagged_diagonals_shape;

        diagonal_offsets_shape(num_jagged_diagonals_shape+1);
        column_indices(num_entries);
        values(num_entries);
        permutations.resize(num_rows);

        row_offsets_block(nb_row+1);
        column_indices_block(num_blocks);

    }

    /*! Construct a \p jad_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    jad_block_matrix(const MatrixType& matrix);

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows, size_t num_cols, size_t num_entries,
                     size_t num_jagged_diagonals_shape, size_t block_size = 2000)
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
        this->num_jagged_diagonals_shape = num_jagged_diagonals_shape;

        diagonal_offsets_shape.resize(num_jagged_diagonals_shape+1);
        column_indices.resize(num_entries);
        values.resize(num_entries);
        permutations.resize(num_rows);

        row_offsets_block.resize(nb_row+1);
        column_indices_block.resize(num_blocks);

    }

    /*! Swap the contents of two \p jad_block_matrix objects.
     *
     *  \param matrix Another \p jad_block_matrix with the same IndexType and ValueType.
     */
    void swap(jad_block_matrix& matrix)
    {
        Parent::swap(matrix);
        diagonal_offsets_shape.swap(matrix.diagonal_offsets_shape);

        column_indices.swap(matrix.column_indices);

        values.swap(matrix.values);
        permutations.swap(matrix.permutations);

        row_offsets_block.swap(matrix.row_offsets_block);
        column_indices_block.swap(matrix.column_indices_block);
    }



    template <typename MatrixOrVector1, typename MatrixOrVector2>
    void operator() (MatrixOrVector1& x, MatrixOrVector2& y){
    	lambda::multiply(*this, x, y);
    }


    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    jad_block_matrix& operator=(const MatrixType& matrix);
}; // class jad_matrix

} // end namespace lambda

#include <lambda/detail/jad_block_matrix.inl>

