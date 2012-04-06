/*
 * convert.inl
 *
 *  Created on: Mar 31, 2012
 *      Author: Filippo Squillace
 */
    

#pragma once


#include <lambda/copy.h>
#include <lambda/detail/dispatch/convert.h>

#include <lambda/format.h>

namespace lambda
{
namespace detail
{

// same format
template <typename SourceType, typename DestinationType,
          typename T1>
void convert(const SourceType& src, DestinationType& dst,
             T1, T1)
{
  lambda::copy(src, dst);
}

// different formats
template <typename SourceType, typename DestinationType,
          typename T1, typename T2>
void convert(const SourceType& src, DestinationType& dst,
             T1, T2)
{
  lambda::detail::dispatch::convert(src, dst,
      typename SourceType::memory_space(),
      typename DestinationType::memory_space());
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

