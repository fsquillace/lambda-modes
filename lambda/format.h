/*
 * utils.h
 *
 *  Created on: Mar 28, 2012
 *      Author: Filippo Squillace
 */

#pragma once

#include <cusp/format.h>

namespace lambda{

struct composite_format : public cusp::unknown_format {};
struct jad_block_format : public cusp::unknown_format {};
struct csr_block_format : public cusp::unknown_format {};

}

