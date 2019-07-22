/*!
 *  Copyright (c) 2019 by Contributors
 * \file Use external fbgemm library call.
 */

#include <cpuinfo.h>
#include <dmlc/logging.h>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/QuantUtilsAvx2.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <memory>
#include <random>
#include "fbgemm_utils.h"

#include <sys/types.h>
#include <unistd.h>
#include <chrono>

namespace tvm {
namespace runtime {
using namespace fbgemm;

using packbmatrix = PackBMatrix<std::int8_t, std::int32_t>;
template <>
struct extension_class_info<packbmatrix> {
  static const int code = 19;
};

TVM_REGISTER_EXT_TYPE(packbmatrix);
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace contrib {

using namespace runtime;
using namespace fbgemm;

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_packb")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* pck_b = args[0];
      packbmatrix* B =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(pck_b);
      B->printPackedMatrix("B");
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.print_col_offsets")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      void* co = args[0];
      std::vector<std::int32_t>* coffsts =
          reinterpret_cast<std::vector<std::int32_t>*>(co);
      std::cout << "size of col offsets" << coffsts->size() << std::endl;
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.packB_with_allocated_tensor")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      //This in the input weight matrix which needs to be packed		    
      DLTensor* inputB = args[0];  
      //This is the iraw memory for packed weights
      DLTensor* packedBufferB = args[1];  
      int num_rows = inputB->shape[0];
      int num_cols = inputB->shape[1];
      PackBMatrix<std::int8_t, std::int32_t> *packB;

      //Assume input is not transposed
      static BlockingFactors params;
      if (args.size() > 2) {
        int cntr = 2;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
  
        packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, num_rows, num_cols,
            reinterpret_cast<const std::int8_t*>(inputB->data), num_cols, 
	    reinterpret_cast<std::int8_t*>(packedBufferB->data), 1,
            &params);
        } else {
        packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, num_rows, num_cols,
            reinterpret_cast<const std::int8_t*>(inputB->data), num_cols, 
	    reinterpret_cast<std::int8_t*>(packedBufferB->data), 1);
      }

    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.create_packed_matrix_with_buffer")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      //This in the input weight matrix which needs to be packed		    
      DLTensor* inputB = args[0];  
      int k = inputB->shape[0];
      int n = inputB->shape[1];

      auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
          matrix_op_t::NoTranspose, 
	  256,
	  32,
	  256,
	  32,
	  1,
	  1,
	  32,
          reinterpret_cast<std::int8_t*>(inputB->data)); 
 
    });

//This function does a gemm where the weights have been pre-packed
//This implementation does not depend on opaque pointers
TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.packedgemm_U8S8ACC32")
    .set_body([](TVMArgs args, TVMRetValue* ret) {

      DLTensor* A = args[0];  // M*K quantized uint8 input
      DLTensor* PackedB = args[1];  // K*N quantized int8 input
      DLTensor* C = args[2];  // M*N int32 output

      int m = A->shape[0];
      int n = PackedB->shape[1];
      int k = A->shape[1];

      auto packB = PackBMatrix<std::int8_t, std::int32_t>(
                matrix_op_t::NoTranspose, 
	        256,
	        32,
	        256,
	        32,
	        1,
	        1,
	        32,
                reinterpret_cast<std::int8_t*>(PackedB->data)); 

      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(A->data), k, nullptr, 1);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, packB, reinterpret_cast<std::int32_t*>(C->data),
                  reinterpret_cast<std::int32_t*>(C->data), n, memcopyObj, 0, 1); 

 });	

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.pack_matrixB_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* W = args[0];  // N*K quantized int8 weight
      int threads = args[1];

      CHECK_EQ(W->ndim, 2);

      int k = W->shape[0];
      int n = W->shape[1];

      BlockingFactors params;
      if (args.size() > 2) {
        int cntr = 2;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];
  
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
            &params);
       //packB->printPackedMatrix("packingB");
        *ret = packB;
      } else {
        auto packB = new PackBMatrix<std::int8_t, std::int32_t>(
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1);
        *ret = packB;
      }
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.compute_col_offsets_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* W = args[0];  // N*K quantized int8 weight
      int threads = args[1];
      std::int32_t w_zero_point = args[2];

      CHECK_EQ(W->ndim, 2);
      int n = W->shape[0];
      int k = W->shape[1];

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      std::vector<std::int32_t>* column_offsets_ =
          new std::vector<std::int32_t>;
      ComputeColumnOffsets<std::int8_t>(
          k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
          *column_offsets_);
      *ret = column_offsets_;
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input

      std::uint64_t wt = args[1];
      void* weight = reinterpret_cast<void*>(static_cast<uint64_t>(wt));
      packbmatrix* packB =
          reinterpret_cast<PackBMatrix<std::int8_t, std::int32_t>*>(weight);

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      //CHECK_EQ(X->ndim, 2);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->ndim, 2);
      // CHECK_EQ(X->shape[1], W->shape[1]);
      //CHECK_EQ(B->ndim, 1);
      // TODO: Ensure correctness here
      // CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      //std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //   reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);

     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];


      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1, &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params); 

	} /*else{

      PackAWithRowOffset<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, *packB, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads); 
     }*/ 
});

bool isValid(BlockingFactors *param)
{
   if (param->MCB % param->MR)
     return false;
   if (param->NCB % param->NR)
     return false;
   if (fbgemmHasAvx512Support()) {
     if (param->MR * (param->NCB / param->NR) > 24)
       return false;
   } else if (fbgemmHasAvx2Support()) {
     if (param->MR * (param->NCB / param->NR) > 16)
       return false;
   }

   return true;

}


TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.gemmint8acc32packedwt_for_tuning")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // K*N quantized uint8 weight

      DLTensor* B = args[2];  // N quantized int8 bias
      DLTensor* Y = args[3];
      int threads = args[9];

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = Y->shape[1];
      int k = X->shape[1];

      // For tuning these will be garbage values
      std::uint64_t co = args[8];
      //void* col_offsts = reinterpret_cast<void*>(static_cast<uint64_t>(co));
      //std::vector<std::int32_t>* column_offsets_ =
      //    reinterpret_cast<std::vector<std::int32_t>*>(col_offsts);
    
     BlockingFactors params;
     if(args.size() > 10){
        int cntr = 10;
        params.MCB = args[cntr];
        params.NCB = args[cntr + 1];
        params.KCB = args[cntr + 2];
        params.MR = args[cntr + 3];
        params.NR = args[cntr + 4];
        params.NR_MIN = args[cntr + 5];
        params.ROW_INTERLEAVE = args[cntr + 6];

        assert (isValid(&params) == true  && "incorrect configuration");
        

        static PackBMatrix<std::int8_t, std::int32_t> packB_ (
            matrix_op_t::NoTranspose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), n, nullptr, 1,
            &params);

      PackAMatrix<std::uint8_t> packA(
          matrix_op_t::NoTranspose, m, k,
          reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1, 
          &params);

      DoNothing<std::int32_t, std::int32_t> doNothing32BitObj;
      memCopy<> memcopyObj (doNothing32BitObj);

      fbgemmPacked(packA, packB_, reinterpret_cast<std::int32_t*>(Y->data),
                  reinterpret_cast<std::int32_t*>(Y->data), n, memcopyObj, 0,
                   threads, &params); 
	}
    });

TVM_REGISTER_GLOBAL("tvm.contrib.fbgemm.fully_connected_int8")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* X = args[0];  // M*K quantized int8 input
      DLTensor* W = args[1];  // N*K quantized int8 weight
      DLTensor* B = args[2];  // N quantized int8 bias
      // ignore the axis and axis_w now for testing purpose
      DLTensor* Y = args[3];
      int threads = args[8];

      CHECK_EQ(X->ndim, 2);
      CHECK_EQ(W->ndim, 2);
      CHECK_EQ(B->ndim, 1);
      CHECK_EQ(X->shape[1], W->shape[1]);
      CHECK_EQ(W->shape[0], B->shape[0]);

      float ReQuant_multiplier = (double)args[7];
      std::int32_t x_zero_point = args[4];
      std::int32_t w_zero_point = args[5];
      std::int32_t y_zero_point = args[6];

      int m = X->shape[0];
      int n = W->shape[0];
      int k = X->shape[1];

      BlockingFactors params;

      if (args.size() > 9) {
        params.MCB = args[9];
        params.NCB = args[10];
        params.KCB = args[11];
        params.MR = args[12];
        params.NR = args[13];
        params.NR_MIN = args[14];
        params.ROW_INTERLEAVE = args[15];
      }

      std::vector<std::int32_t> row_offsets_(
          PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());
      std::vector<std::int32_t> Y_int32_(n * m);
      std::vector<std::int32_t> column_offsets_;

      std::vector<TensorQuantizationParams> temp_qparams;
      temp_qparams.push_back(TensorQuantizationParams{1.0, w_zero_point});

      if (args.size() > 9) {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::Transpose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1,
            &params);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data(), &params);

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0, threads,
                     &params);  // num_threads

      } else {
        PackBMatrix<std::int8_t, std::int32_t> packB(
            matrix_op_t::Transpose, k, n,
            reinterpret_cast<const std::int8_t*>(W->data), k, nullptr, 1);

        PackAWithRowOffset<std::uint8_t> packA(
            matrix_op_t::NoTranspose, m, k,
            reinterpret_cast<const std::uint8_t*>(X->data), k, nullptr, 1,
            row_offsets_.data());

        ComputeColumnOffsets<std::int8_t>(
            k, n, reinterpret_cast<const std::int8_t*>(W->data), temp_qparams,
            column_offsets_);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false> outputProcObj(
            doNothingObj, &ReQuant_multiplier, y_zero_point, x_zero_point,
            &w_zero_point, packA.getRowOffsetBuffer(), column_offsets_.data(),
            reinterpret_cast<const std::int32_t*>(B->data), n);

        fbgemmPacked(packA, packB, reinterpret_cast<std::uint8_t*>(Y->data),
                     Y_int32_.data(), n, outputProcObj, 0,
                     threads);  // num_threads
      }
    });

}  // namespace contrib
}  // namespace tvm
