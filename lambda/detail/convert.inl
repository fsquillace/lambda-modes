/*
 * convert.inl
 *
 *  Created on: Mar 31, 2012
 *      Author: Filippo Squillace
 */
    

#pragma once

//#include <cusp/detail/copy.inl>
#include <cusp/copy.h>


#include <lambda/format.h>

namespace lambda
{
namespace detail
{
// There is no conversion for composite_matrix with different format
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst,
             lambda::composite_format, lambda::composite_format)
{
	cusp::detail::copy_matrix_dimensions(src, dst);
	cusp::copy(src.M11, dst.M11);
	cusp::copy(src.M11, dst.M12);

	cusp::copy(src.M11, dst.L11);
	cusp::copy(src.M11, dst.L21);
	cusp::copy(src.M11, dst.L22);

}
  

} // end namespace detail

/////////////////
// Entry Point //
/////////////////
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst)
{

  lambda::detail::convert(src, dst,
      typename SourceType::format(),
      typename DestinationType::format());
}

} // end namespace lambda

