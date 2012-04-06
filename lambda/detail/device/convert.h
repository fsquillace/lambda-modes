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

#include <lambda/format.h>

namespace lambda
{
namespace detail
{
namespace device
{

/////////
// Composite //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             lambda::composite_format,
             lambda::composite_format)
{
	// NOT implemented

}



/////////////////
// Entry Point //
/////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    lambda::detail::device::convert(src, dst,
            typename Matrix1::format(),
            typename Matrix2::format());
}
            
} // end namespace device
} // end namespace detail
} // end namespace lambda

