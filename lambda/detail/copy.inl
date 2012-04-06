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

#include <lambda/format.h>


#include <cusp/copy.h>


namespace lambda
{
namespace detail
{

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          lambda::composite_format,
          lambda::composite_format)
{
	cusp::detail::copy_matrix_dimensions(src, dst);
	cusp::copy(src.M11, dst.M11);
	cusp::copy(src.M11, dst.M12);

	cusp::copy(src.M11, dst.L11);
	cusp::copy(src.M11, dst.L21);
	cusp::copy(src.M11, dst.L22);


}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          lambda::jad_block_format,
          lambda::jad_block_format)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries, src.num_jagged_diagonals_shape, src.block_size);

    cusp::copy(src.permutations, dst.permutations);
    cusp::copy(src.diagonal_offsets_shape, dst.diagonal_offsets_shape);
    cusp::copy(src.values, dst.values);
    cusp::copy(src.column_indices, dst.column_indices);

    cusp::copy(src.row_offsets_block, dst.row_offsets_block);
    cusp::copy(src.column_indices_block, dst.column_indices_block);

}

}
/////////////////
// Entry Point //
/////////////////

template <typename T1, typename T2>
void copy(const T1& src, T2& dst)
{
  lambda::detail::copy(src, dst, typename T1::format(), typename T2::format());
}

} // end namespace lambda

