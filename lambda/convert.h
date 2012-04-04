/*
 * convert.h
 *
 *  Created on: Mar 31, 2012
 *      Author: Filippo Squillace
 */

#pragma once


namespace lambda
{

/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p copy : Convert between matrix formats
 *
 * \note DestinationType will be resized as necessary
 *
 * \see \p lambda::copy
 */
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst);

/*! \}
 */

} // end namespace lambda

#include <lambda/detail/convert.inl>

