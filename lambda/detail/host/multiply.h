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


namespace lambda
{
namespace detail
{
namespace host
{

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              lambda::composite_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
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
  lambda::detail::host::multiply(A, B, C,
                               typename Matrix::format(),
                               typename MatrixOrVector1::format(),
                               typename MatrixOrVector2::format());
}

} // end namespace host
} // end namespace detail
} // end namespace lambda

