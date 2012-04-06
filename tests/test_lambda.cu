/*
 * test_lambda.cu
 *
 *  Created on: Mar 28, 2012
 *      Author: Filippo Squillace
 */


#include <iostream>
#include <sstream>
#include <string.h>


#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/krylov/arnoldi.h>
#include <cusp/detail/matrix_base.h>
//#include "../../cusp/krylov/arnoldi.h"


#include <lambda/composite_matrix.h>

#include <cuspla.cu>


#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestFixture.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestSuite.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>


void checkStatus(culaStatus status)
{
	if(!status)
		return;
	if(status == culaArgumentError)
		printf("Invalid value for parameter %d\n", culaGetErrorInfo());
	else if(status == culaDataError)
		printf("Data error (%d)\n", culaGetErrorInfo());
	else if(status == culaBlasError)
		printf("Blas error (%d)\n", culaGetErrorInfo());
	else if(status == culaRuntimeError)
		printf("Runtime error (%d)\n", culaGetErrorInfo());
	else
		printf("%s\n", culaGetStatusString(status));

	culaShutdown();
	exit(EXIT_FAILURE);
}


class LambdaTestCase : public CppUnit::TestFixture {

	CPPUNIT_TEST_SUITE (LambdaTestCase);
	CPPUNIT_TEST (test_host_multiply);
	CPPUNIT_TEST (test_device_multiply);
	CPPUNIT_TEST(test_host_arnoldi);
	CPPUNIT_TEST(test_device_arnoldi);
	CPPUNIT_TEST(test_host_iram);
	CPPUNIT_TEST_SUITE_END ();

	typedef int    IndexType;
	typedef float ValueType;
	typedef cusp::array2d<float,cusp::device_memory, cusp::column_major> DeviceMatrix_array2d;
	typedef cusp::array2d<float, cusp::host_memory, cusp::column_major>   HostMatrix_array2d;

	typedef cusp::array1d<float,cusp::device_memory> DeviceVector_array1d;
	typedef cusp::array1d<float, cusp::host_memory>   HostVector_array1d;

	typedef cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>   HostMatrix_csr;
	typedef cusp::csr_matrix<IndexType, ValueType, cusp::device_memory>   DeviceMatrix_csr;

	typedef lambda::composite_matrix<IndexType, ValueType, cusp::host_memory, HostMatrix_csr>   HostMatrix_comp;
	typedef lambda::composite_matrix<IndexType, ValueType, cusp::device_memory,  DeviceMatrix_csr> DeviceMatrix_comp;


private:
	DeviceMatrix_comp dev_mat;
	HostMatrix_comp host_mat;


public:

	void setUp()
	{

		culaStatus status;
		status = culaInitialize();
		checkStatus(status);

		std::string path = "data/positive-definite/lehmer20.mtx";


		HostMatrix_csr M11, M12, L11, L21, L22;
		cusp::io::read_matrix_market_file(M11, path);
		cusp::io::read_matrix_market_file(M12, path);
		cusp::io::read_matrix_market_file(L11, path);
		cusp::io::read_matrix_market_file(L21, path);
		cusp::io::read_matrix_market_file(L22, path);

		host_mat = HostMatrix_comp(M11,M12,L11,L21,L22);

		dev_mat = DeviceMatrix_comp(host_mat);

	}

	void tearDown()
	{
		culaShutdown();
	}

	void test_host_multiply(){

		HostVector_array1d y1(host_mat.num_rows);
		HostVector_array1d x(host_mat.num_cols, ValueType(1));
		cusp::multiply(host_mat,x,y1);


		// ******** TESTING *************

		// A = L11^{-1}*(M_{11}+M_{12}*L_{22}^{-1}*L_{21})
		HostMatrix_array2d L11_inv, L22_inv, L21, M11, M12;

		cusp::convert(host_mat.L11, L11_inv);
		cuspla::getri(L11_inv);
		cusp::convert(host_mat.L22, L22_inv);
		cuspla::getri(L22_inv);
		cusp::convert(host_mat.L21, L21);
		cusp::convert(host_mat.M11, M11);
		cusp::convert(host_mat.M12, M12);

		HostMatrix_array2d tmp1, tmp2;
		cuspla::gemm(M12, L22_inv, tmp1, ValueType(1));
		cuspla::gemm(tmp1, L21, tmp2, ValueType(1));

		cusp::blas::axpy(M11.values.begin() , M11.values.end(), tmp2.values.begin(), ValueType(1));
		HostMatrix_array2d A;
		cuspla::gemm(L11_inv, tmp2, A, ValueType(1));

		HostVector_array1d y2(host_mat.num_rows);
		cuspla::gemv(A,x,y2);

		ValueType errRel = nrmVector("host_multiply: ", y1, y2);
		CPPUNIT_ASSERT( errRel < 1.0e-3 );
	}

	void test_device_multiply(){

		DeviceVector_array1d y1(dev_mat.num_rows);
		DeviceVector_array1d x(dev_mat.num_cols, ValueType(1));
		cusp::multiply(dev_mat,x,y1);
		HostVector_array1d y1_host(host_mat.num_rows);
		cusp::copy(y1, y1_host);
		HostVector_array1d x_host(host_mat.num_rows);
		cusp::copy(x, x_host);



		// ******** TESTING *************

		// A = L11^{-1}*(M_{11}+M_{12}*L_{22}^{-1}*L_{21})
		HostMatrix_array2d L11_inv, L22_inv, L21, M11, M12;

		cusp::convert(host_mat.L11, L11_inv);
		cuspla::getri(L11_inv);
		cusp::convert(host_mat.L22, L22_inv);
		cuspla::getri(L22_inv);
		cusp::convert(host_mat.L21, L21);
		cusp::convert(host_mat.M11, M11);
		cusp::convert(host_mat.M12, M12);

		HostMatrix_array2d tmp1, tmp2;
		cuspla::gemm(M12, L22_inv, tmp1, ValueType(1));
		cuspla::gemm(tmp1, L21, tmp2, ValueType(1));

		cusp::blas::axpy(M11.values.begin() , M11.values.end(), tmp2.values.begin(), ValueType(1));
		HostMatrix_array2d A;
		cuspla::gemm(L11_inv, tmp2, A, ValueType(1));

		HostVector_array1d y2(host_mat.num_rows);
		cuspla::gemv(A,x_host,y2);


		ValueType errRel = nrmVector("device_multiply: ", y1_host, y2);
		CPPUNIT_ASSERT( errRel < 1.0e-3 );
	}


	void test_host_arnoldi()
	{

		size_t m = 10;
		HostMatrix_array2d H(m, m);
		HostMatrix_array2d V(host_mat.num_rows, m);
		HostVector_array1d f(host_mat.num_rows, ValueType(0));

		cusp::krylov::arnoldi(host_mat, H, V, f, 0, 3);
		cusp::krylov::arnoldi(host_mat, H, V, f, 2, 5);
		cusp::krylov::arnoldi(host_mat, H, V, f, 4, m);



		// ******* TESTING ***********

		HostMatrix_array2d A2d;
		HostMatrix_array2d V2;
		HostMatrix_array2d H2;

		HostMatrix_array2d C;
		HostMatrix_array2d C2;


		size_t N = host_mat.num_rows;

		// A = L11^{-1}*(M_{11}+M_{12}*L_{22}^{-1}*L_{21})
		HostMatrix_array2d L11_inv, L22_inv, L21, M11, M12;

		cusp::convert(host_mat.L11, L11_inv);
		cuspla::getri(L11_inv);
		cusp::convert(host_mat.L22, L22_inv);
		cuspla::getri(L22_inv);
		cusp::convert(host_mat.L21, L21);
		cusp::convert(host_mat.M11, M11);
		cusp::convert(host_mat.M12, M12);

		HostMatrix_array2d tmp1, tmp2;
		cuspla::gemm(M12, L22_inv, tmp1, ValueType(1));
		cuspla::gemm(tmp1, L21, tmp2, ValueType(1));

		cusp::blas::axpy(M11.values.begin() , M11.values.end(), tmp2.values.begin(), ValueType(1));
		cuspla::gemm(L11_inv, tmp2, A2d, ValueType(1));


		// create submatrix V2
		cusp::copy(V, V2);
		V2.resize(N,m);

		// create submatrix H2
		H2.resize(m,m);
		size_t l = H.num_rows;
		for(size_t j=0; j<m; j++)
			thrust::copy(H.values.begin()+ l*j, H.values.begin()+ l*j +m, H2.values.begin()+ m*j);

		cusp::multiply(A2d, V2, C);

		cusp::multiply(V2, H2, C2);

		cusp::blas::axpy(f.begin() , f.end(), C2.values.begin()+(m-1)*N, ValueType(1));


		ValueType errRel = nrmVector("host_arnoldi: ", C.values, C2.values);
		CPPUNIT_ASSERT( errRel < 1.0e-3 );

	}

	void test_device_arnoldi()
	{

		size_t m = 10;
		DeviceMatrix_array2d H(m, m);
		DeviceMatrix_array2d V(dev_mat.num_rows, m);
		DeviceVector_array1d f(dev_mat.num_rows, ValueType(0));

		//		  DeviceMatrix_csr dev_mat;
		//		  cusp::convert(dev_mat_def_pos[i], dev_mat);
		cusp::krylov::arnoldi(dev_mat, H, V, f, 0, 3);
		cusp::krylov::arnoldi(dev_mat, H, V, f, 2, 5);
		cusp::krylov::arnoldi(dev_mat, H, V, f, 4, m);


		// ******* TESTING ***********

		HostMatrix_array2d A2d;
		HostMatrix_array2d V2;
		HostMatrix_array2d H2;

		HostMatrix_array2d C;
		HostMatrix_array2d C2;
		HostVector_array1d f_host;
		cusp::convert(f,f_host);

		size_t N = host_mat.num_rows;

		// A = L11^{-1}*(M_{11}+M_{12}*L_{22}^{-1}*L_{21})
		HostMatrix_array2d L11_inv, L22_inv, L21, M11, M12;

		cusp::convert(host_mat.L11, L11_inv);
		cuspla::getri(L11_inv);
		cusp::convert(host_mat.L22, L22_inv);
		cuspla::getri(L22_inv);
		cusp::convert(host_mat.L21, L21);
		cusp::convert(host_mat.M11, M11);
		cusp::convert(host_mat.M12, M12);

		HostMatrix_array2d tmp1, tmp2;
		cuspla::gemm(M12, L22_inv, tmp1, ValueType(1));
		cuspla::gemm(tmp1, L21, tmp2, ValueType(1));

		cusp::blas::axpy(M11.values.begin() , M11.values.end(), tmp2.values.begin(), ValueType(1));
		cuspla::gemm(L11_inv, tmp2, A2d, ValueType(1));



		// create submatrix V2
		cusp::copy(V, V2);
		V2.resize(N,m);

		// create submatrix H2
		H2.resize(m,m);
		size_t l = H.num_rows;
		for(size_t j=0; j<m; j++)
			thrust::copy(H.values.begin()+ l*j, H.values.begin()+ l*j +m, H2.values.begin()+ m*j);

		cusp::multiply(A2d, V2, C);

		cusp::multiply(V2, H2, C2);

		cusp::blas::axpy(f_host.begin() , f_host.end(), C2.values.begin()+(m-1)*N, float(1));



		ValueType errRel = nrmVector("device_arnoldi: ", C.values, C2.values);
		CPPUNIT_ASSERT( errRel < 1.0e-3 );

	}


	void test_host_iram(){ //  TODO test iram with composite matrix
		size_t k = 4;

		size_t n = host_mat.num_rows;
		size_t m = host_mat.num_cols;
		HostMatrix_array2d eigvects;
		HostMatrix_array2d A2d;
		HostVector_array1d eigvals;
		HostVector_array1d y1, eigvec(m);

		cusp::krylov::implicitly_restarted_arnoldi(host_mat,\
				eigvals, eigvects, k, 0);

		// A = L11^{-1}*(M_{11}+M_{12}*L_{22}^{-1}*L_{21})
		HostMatrix_array2d L11_inv, L22_inv, L21, M11, M12;

		cusp::convert(host_mat.L11, L11_inv);
		cuspla::getri(L11_inv);
		cusp::convert(host_mat.L22, L22_inv);
		cuspla::getri(L22_inv);
		cusp::convert(host_mat.L21, L21);
		cusp::convert(host_mat.M11, M11);
		cusp::convert(host_mat.M12, M12);

		HostMatrix_array2d tmp1, tmp2;
		cuspla::gemm(M12, L22_inv, tmp1, ValueType(1));
		cuspla::gemm(tmp1, L21, tmp2, ValueType(1));

		cusp::blas::axpy(M11.values.begin() , M11.values.end(), tmp2.values.begin(), ValueType(1));
		cuspla::gemm(L11_inv, tmp2, A2d, ValueType(1));


		for(size_t j=0; j<eigvals.size(); j++){
			thrust::copy(eigvects.values.begin()+ j*n, eigvects.values.begin()+ (j+1)*n,eigvec.begin());
			cuspla::gemv(A2d, eigvec, y1, false);
			cusp::blas::scal(eigvec, (ValueType)eigvals[j]);

			std::stringstream j_str, eigval_str;
			j_str << j;
			eigval_str << eigvals[j];

			ValueType errRel = nrmVector("host_iram eigval["+j_str.str()+"]:"+eigval_str.str(), y1, eigvec);
			CPPUNIT_ASSERT( errRel < 1.0e-2 );
		}
	}


//	test_device_iram(){
//		size_t k = 4;
//
//		size_t n = host_mat.num_rows;
//		size_t m = host_mat.num_cols;
//		DeviceMatrix_array2d eigvects;
//		HostMatrix_array2d A2d;
//		DeviceVector_array1d eigvals;
//		HostVector_array1d y1, eigvec(m);
//
//		cusp::krylov::implicitly_restarted_arnoldi(dev_mat,\
//				eigvals, eigvects, k, 0);
//
//
//		cusp::convert(dev_mat, A2d);
//
//		for(size_t j=0; j<eigvals.size(); j++){
//			thrust::copy(eigvects.values.begin()+ j*n, eigvects.values.begin()+ (j+1)*n,eigvec.begin());
//			cuspla::gemv(A2d, eigvec, y1, false);
//			cusp::blas::scal(eigvec, (ValueType)eigvals[j]);
//
//			std::stringstream j_str, eigval_str;
//			j_str << j;
//			eigval_str << eigvals[j];
//
//			ValueType errRel = nrmVector("host_iram eigval["+j_str.str()+"]:"+eigval_str.str(), y1, eigvec);
//			CPPUNIT_ASSERT( errRel < 1.0e-2 );
//
//		}
//	}



	template <typename Array1d>
	ValueType nrmVector(std::string title, Array1d& A, Array1d& A2){
		ValueType nrmA = cusp::blas::nrm2(A);
		ValueType nrmA2 = cusp::blas::nrm2(A2);
		// Calculates the difference and overwrite the matrix C
		cusp::blas::axpy(A, A2, ValueType(-1));
		ValueType nrmDiff = cusp::blas::nrm2(A2);



		ValueType errRel = ValueType(0);
		if(nrmA==ValueType(0))
			errRel = ValueType(1.0e-30);
		else
			errRel = nrmDiff/nrmA;

#ifdef VERBOSE
#ifndef VVERBOSE
		if(errRel != errRel || errRel >= 1.0e-2){ // Checks if error is nan
#endif VVERBOSE

			std::cout << title << ": AbsoluteErr=" << nrmDiff <<\
					" RelativeErr=" << errRel << "\n" << std::endl;
#ifndef VVERBOSE
		}
#endif VVERBOSE
#endif


		return errRel;
	}



};





CPPUNIT_TEST_SUITE_REGISTRATION( LambdaTestCase );

int main(int argc, char** argv)
{

	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest( registry.makeTest() );
	runner.run();
	return 0;

}



