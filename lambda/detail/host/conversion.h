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

#include <cusp/ell_matrix.h>
#include <cusp/exception.h>

#include <cusp/detail/host/conversion_utils.h>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/count.h>


#include <cusp/detail/host/convert.h>
#include <stdio.h>

#include <assert.h>

#include <cusp/detail/host/multiply.h>

#include <cusp/convert.h>
#include <lambda/csr_block_matrix.h>

namespace lambda
{
namespace detail
{
namespace host
{


template <typename Matrix1, typename Matrix2>
void csr_to_csr_block(const Matrix1& src, Matrix2& dst,
                const size_t block_size)
{

    // Note: we suppose that the pattern appears in the first block on up-left side of the matrix

    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // Determine the num of blocks
    size_t nb_col = (src.num_cols/block_size);
    nb_col += (src.num_cols % block_size==0)?0:1;
    size_t nb_row = (src.num_rows/block_size);
    nb_row += (src.num_rows % block_size==0)?0:1;

    size_t num_blocks = nb_col * nb_row;

//    size_t num_entries_per_block = 0;
//    for(size_t i = 0; i < block_size; i++){
//        for(IndexType p =src.row_offsets[i]; p<src.row_offsets[i+1]; p++ ){
//            IndexType j = src.column_indices[p];
//            // Check out the i and j block
//            size_t i_block = (i/block_size);
//            size_t j_block = (j/block_size);
//            size_t id_block = i_block*nb_col + j_block;
//            if(id_block==0){
//                num_entries_per_block++;
////                printf("element in i:%d j:%d id_block:%d\n", i, j, id_block);
//            }
//        }
//    }


    dst.resize(src.num_rows, src.num_cols, src.num_entries, block_size);

    printf("num_blocks=%d nb_col=%d nb_row=%d block_size=%d\n",dst.num_blocks, dst.nb_col,
           dst.nb_row, dst.block_size);

    // Shape information
    size_t counter = 0;
    for(size_t i = 0; i < block_size; i++){
        dst.row_offsets_shape[i] = counter;

        for(IndexType p =src.row_offsets[i]; p<src.row_offsets[i+1]; p++ ){
            IndexType j = src.column_indices[p];
            // Check out the i and j block
            size_t i_block = (i/block_size);
            size_t j_block = (j/block_size);
            size_t id_block = i_block*nb_col + j_block;
            if(id_block==0){
                // Add the shape info into the data structures
                dst.column_indices_shape[counter] = j;
//                printf("element in i:%d j:%d --> roff:%d\n", i, j, dst.row_offsets_block[i]);
                counter++;
            }
        }
    }
//    assert(counter==num_entries_per_block);
    dst.column_indices_shape.resize(counter);
    dst.row_offsets_shape[block_size] = counter;
    printf("not_null_entries_per_block:%d\n", counter);



    // Blocks information
    cusp::array1d<bool, cusp::host_memory> exists_pattern(num_blocks);
    for(size_t id_b=0; id_b<num_blocks; id_b++){
        exists_pattern[id_b] = false;
    }

    for(size_t i = 0; i < src.num_rows; i++){
        for(IndexType p =src.row_offsets[i]; p<src.row_offsets[i+1]; p++ ){
            IndexType j = src.column_indices[p];
            // Check out the i and j block
            size_t i_block = (i/block_size);
            size_t j_block = (j/block_size);
            size_t id_block = i_block*nb_col + j_block;
            exists_pattern[id_block] = true;
//            printf("exists pattern in block[%d]\n", id_block);

        }
    }
    counter = 0;
    for(size_t i = 0; i < nb_row; i++){
        dst.row_offsets_block[i]=counter;
//        printf("row_offset_block[%d]:%d\n", i, counter);
        for(size_t j = 0; j < nb_col; j++){
            if(exists_pattern[i*nb_col + j]){
                dst.column_indices_block[counter] = j;
//                printf("column_indices_block[%d]:%d\n", counter, j);
                counter++;

            }
        }
    }
    dst.row_offsets_block[nb_row]=counter;
    printf("not_null_blocks:%d\n", counter);
    dst.column_indices_block.resize(counter);


    // Values information
//    cusp::copy(src.values, dst.values);

    counter=0;
    for(size_t id_b=0; id_b<num_blocks; id_b++){
        if(not exists_pattern[id_b])
            continue;

        for(size_t i = 0; i < src.num_rows; i++){
            for(IndexType p =src.row_offsets[i]; p<src.row_offsets[i+1]; p++ ){
                IndexType j = src.column_indices[p];
                // Check out the i and j block
                size_t i_block = (i/block_size);
                size_t j_block = (j/block_size);
                size_t id_block = i_block*nb_col + j_block;
                if(id_b==id_block){
                    dst.values[counter] = src.values[p];
//                    printf("pos:%d --> i:%d j:%d val:%f\n", counter, i, j, src.values[p]);
                    counter++;
                }
            }
        }
    }
}

void merge(int* vec, int* vec2, int start, int med, int stop){
    int* a = new int[stop-start+1];
    int* a2 = new int[stop-start+1];

    int i1 = start; int i2 = med+1; int i3 = 0;
    for (; (i1 <= med)&&(i2 <= stop); i3++ )
        if ( vec[i1]>vec[i2]){
            a[i3]=vec[i1];
            a2[i3]=vec2[i1];
            i1++;
        }
        else{
            a[i3]=vec[i2];
            a2[i3]=vec2[i2];
            i2++;
        }
    for ( ; i1 <= med; i1++, i3++ ){
        a[i3] = vec[i1];
        a2[i3] = vec2[i1];
    }
    for ( ; i2 <= stop; i2++, i3++ ){
        a[i3] = vec[i2];
        a2[i3] = vec2[i2];
    }
    for ( i3 = 0, i1 = start; i1 <= stop; i3++, i1++ ){
        vec[i1] = a[i3];
        vec2[i1] = a2[i3];
//        printf("merge: vec[%d]:%d start:%d med:%d stop:%d\n", i1, vec[i1], start, med, stop);
    }

    delete [] a;
    delete [] a2;

}

void mergesort(int* vec, int* vec2, int start, int stop){
    if(stop<=start){
        return;
    }
    int med = start + (stop-start)/2;
    mergesort(vec, vec2, start, med);
    mergesort(vec, vec2, med+1, stop);
    merge(vec, vec2, start, med, stop);

}





template <typename Matrix1, typename Matrix2>
void csr_to_jad_block(const Matrix1& src, Matrix2& dst,
                const size_t block_size)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // Determine the num of blocks
    size_t nb_col = (src.num_cols/block_size);
    nb_col += (src.num_cols % block_size==0)?0:1;
    size_t nb_row = (src.num_rows/block_size);
    nb_row += (src.num_rows % block_size==0)?0:1;

//    size_t num_blocks = nb_col * nb_row;


printf("begin: calculate permutation\n");
printf("block_size:%d\n", block_size);
    // Permutation of src csr_jagged=f(src)
    cusp::array1d<IndexType, cusp::host_memory> num_entries_vec(block_size);
    cusp::array1d<IndexType, cusp::host_memory> permutations(block_size);

    for(int i=0; i<block_size; i++){
        num_entries_vec[i] = src.row_offsets[i+1] - src.row_offsets[i];
        permutations[i] = i;
//        printf("num_entries_vec[%d]:%d\n", i, num_entries_vec[i]);
    }

    mergesort(thrust::raw_pointer_cast(&num_entries_vec[0]), thrust::raw_pointer_cast(&permutations[0]),
              0, block_size-1);

//    for(int i=0; i<src.num_rows; i++){
//        printf("num_entries_vec[%d]:%d\n", i, num_entries_vec[i]);
//    }
//    for(int i=0; i<src.num_rows; i++){
//        printf("permutations[%d]:%d\n", i, permutations[i]);
//    }

    // Replicate the permutation for all the matrix
    permutations.resize(src.num_rows);
    for(int i_b=1; i_b<nb_row; i_b++){
        for(int i=0; i<block_size; i++){
            permutations[i_b*block_size + i]= i_b*block_size + permutations[i];
        }
    }

//    for(int i=0; i<src.num_rows; i++){
//        printf("permutations[%d]:%d\n", i, permutations[i]);
//    }

printf("end: calculate permutation\n");

printf("begin: csr_jagged\n");
    // build the csr permutated matrix
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> csr_jagged;
    csr_jagged.resize(src.num_rows, src.num_cols, src.num_entries);

    size_t sum_lenght = 0;
    for(int i=0; i<src.num_rows; i++){
        csr_jagged.row_offsets[i] = sum_lenght;
//        printf("row_offsets[%d]:%d\n", i, csr_jagged.row_offsets[i]);
        sum_lenght += src.row_offsets[permutations[i]+1] - src.row_offsets[permutations[i]];
    }
    csr_jagged.row_offsets[src.num_rows] = sum_lenght; // The last

    for(int i=0; i<src.num_rows; i++){
        IndexType start_src = src.row_offsets[permutations[i]];
        IndexType stop_src = src.row_offsets[permutations[i]+1];

        IndexType start_jad = csr_jagged.row_offsets[i];
        IndexType stop_jad = csr_jagged.row_offsets[i+1];

        for(IndexType jj_src=start_src, jj_jad=start_jad; jj_src<stop_src && jj_jad<stop_jad; jj_src++, jj_jad++){

            csr_jagged.values[jj_jad] = src.values[jj_src];
            csr_jagged.column_indices[jj_jad] = src.column_indices[jj_src];

//            printf("values[%d]:%f\n", jj_jad, csr_jagged.values[jj_jad]);
//            printf("column_indices[%d]:%d\n", jj_jad, csr_jagged.column_indices[jj_jad]);

        }
    }
//    for(int i=0; i<csr_jagged.num_rows; i++){
//        printf("csr_jagged.row_offsets[%d]:%d\n", i, csr_jagged.row_offsets[i]);
//        for(int jj=csr_jagged.row_offsets[i]; jj<csr_jagged.row_offsets[i+1]; jj++){
//            printf("csr_jagged.column_indices[%d]:%d\n", jj, csr_jagged.column_indices[jj]);
//            printf("csr_jagged.values[%d]:%f\n", jj, csr_jagged.values[jj]);
//        }
//    }
printf("end: csr_jagged\n");
printf("begin: csr_block\n");

    // convert HostMatrix to TestMatrix on host
    lambda::csr_block_matrix<IndexType, ValueType, cusp::host_memory> csr_block_jagged;
    // CSR -> HYB BLOCK
    lambda::detail::host::csr_to_csr_block(csr_jagged, csr_block_jagged, block_size);

printf("end: csr_block\n");
printf("begin: dst\n");


    // Calculate the conversion
    size_t num_jagged_diagonals_shape = csr_block_jagged.row_offsets_shape[1] - csr_block_jagged.row_offsets_shape[0];
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_jagged_diagonals_shape, block_size);
    cusp::copy(permutations, dst.permutations);
    cusp::copy(csr_block_jagged.row_offsets_block, dst.row_offsets_block);
    cusp::copy(csr_block_jagged.column_indices_block, dst.column_indices_block);

    size_t num_entries_shape = csr_block_jagged.row_offsets_shape[block_size];
    thrust::fill(dst.column_indices.begin(), dst.column_indices.end(), ValueType(-1));
    thrust::fill(dst.values.begin(),         dst.values.end(),         ValueType(0));

    size_t counter = 0;
    // Initialize diagonal_offsets_shape
    for(size_t d=0; d<num_jagged_diagonals_shape; d++){
        dst.diagonal_offsets_shape[d] = counter;
//        printf("diagonal_offsets_shape[%d]:%d\n",d, counter);
        for(int i=0; i<block_size; i++){
            IndexType lenght_row = csr_block_jagged.row_offsets_shape[i+1] - csr_block_jagged.row_offsets_shape[i];
            if(d<lenght_row){
                counter++;
            }
        }
    }
    dst.diagonal_offsets_shape[num_jagged_diagonals_shape]=counter;
//    printf("diagonal_offsets_shape[%d]:%d\n", num_jagged_diagonals_shape, counter);



    counter=0;
    for(int i_b=0; i_b<nb_row; i_b++){
        IndexType start_row_block = csr_block_jagged.row_offsets_block[i_b];
        IndexType stop_row_block = csr_block_jagged.row_offsets_block[i_b+1];


        for(size_t d=0; d<num_jagged_diagonals_shape; d++){
             for(int i=0; i<block_size; i++){

                IndexType lenght_row = csr_block_jagged.row_offsets_shape[i+1] - csr_block_jagged.row_offsets_shape[i];
                if(d<lenght_row){
                    IndexType jj = csr_block_jagged.row_offsets_shape[i] + d;
                    for(int jj_b=start_row_block; jj_b<stop_row_block; jj_b++){

                        dst.values[counter] = csr_block_jagged.values[num_entries_shape*jj_b+ jj];
                        dst.column_indices[counter] = csr_block_jagged.column_indices_block[jj_b]*block_size+
                                    csr_block_jagged.column_indices_shape[jj];
//                        printf("values[%d]:%f\n", counter, dst.values[counter]);
//                        printf("column_indices[%d]:%d\n", counter, dst.column_indices[counter]);
                        counter++;
                    }
                }
            }
        }
     }



printf("end: dst\n");

printf("num_jagged_diagonals_shape:%d\n", num_jagged_diagonals_shape);


}


} // end namespace host
} // end namespace detail
} // end namespace lambda

