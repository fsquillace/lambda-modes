/*
 * test_jad.cu
 *
 *  Created on: Apr 6, 2011
 *      Author: Squillace Filippo
 */

//#define CUSP_USE_TEXTURE_MEMORY


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

#include <lambda/jad_block_matrix.h>
#include <lambda/convert.h>


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


class JadBlockTestCase : public CppUnit::TestFixture {

	CPPUNIT_TEST_SUITE (JadBlockTestCase);
	CPPUNIT_TEST (test_jad_block);
	CPPUNIT_TEST_SUITE_END ();

	typedef int    IndexType;
	typedef float ValueType;
	typedef cusp::array2d<float,cusp::device_memory, cusp::column_major> DeviceMatrix_array2d;
	typedef cusp::array2d<float, cusp::host_memory, cusp::column_major>   HostMatrix_array2d;

	typedef cusp::array1d<float,cusp::device_memory> DeviceVector_array1d;
	typedef cusp::array1d<float, cusp::host_memory>   HostVector_array1d;

	typedef cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix_csr;
	typedef cusp::csr_matrix<IndexType, float, cusp::host_memory>   HostMatrix_csr;

	typedef lambda::jad_block_matrix<IndexType, ValueType, cusp::host_memory> HostMatrix_jad_block;
    typedef lambda::jad_block_matrix<IndexType, ValueType, cusp::device_memory> DeviceMatrix_jad_block;

private:

	std::vector<std::string> path_def_pos;
	std::vector<DeviceMatrix_csr> dev_mat_def_pos;
	std::vector<HostMatrix_csr> host_mat_def_pos;


public:

	void setUp()
	{

		culaStatus status;
		status = culaInitialize();
		checkStatus(status);


		// ################################ POSITIVE DEFINITE #####################
		path_def_pos = std::vector<std::string>(6);
		path_def_pos[0] = "data/positive-definite/lehmer10.mtx";
		path_def_pos[1] = "data/positive-definite/lehmer20.mtx";
		path_def_pos[2] = "data/positive-definite/lehmer50.mtx";
		path_def_pos[3] = "data/positive-definite/lehmer100.mtx";
		path_def_pos[4] = "data/positive-definite/lehmer200.mtx";
		path_def_pos[5] = "data/positive-definite/moler200.mtx";
//		path_def_pos[0] = "data/L11_4_ringhals.mtx";

		host_mat_def_pos = std::vector<HostMatrix_csr>(path_def_pos.size());
		dev_mat_def_pos = std::vector<DeviceMatrix_csr>(path_def_pos.size());
		for(size_t i=0; i<path_def_pos.size(); i++){
			cusp::io::read_matrix_market_file(host_mat_def_pos[i], path_def_pos[i]);
			dev_mat_def_pos[i] = DeviceMatrix_csr(host_mat_def_pos[i]);
		}


	}

	void tearDown()
	{
		culaShutdown();
	}

	void test_jad_block()
	{
		for(size_t i=0; i<path_def_pos.size(); i++){

			DeviceVector_array1d x(host_mat_def_pos[i].num_cols, ValueType(1)),\
					y1(host_mat_def_pos[i].num_rows);

			HostVector_array1d x_host(host_mat_def_pos[i].num_cols, ValueType(1)),\
					y1_host(host_mat_def_pos[i].num_rows),y2(host_mat_def_pos[i].num_rows);

			// convert HostMatrix to TestMatrix on host
			HostMatrix_jad_block test_matrix_on_host;
			// CSR -> JAD BLOCK
			lambda::convert(host_mat_def_pos[i], test_matrix_on_host);

			// transfer TestMatrix to device
			DeviceMatrix_jad_block test_matrix_on_device(test_matrix_on_host);

			cusp::multiply(test_matrix_on_device, x, y1);
			cusp::copy(y1, y1_host);
			cusp::multiply(host_mat_def_pos[i], x_host, y2);

			ValueType errRel = nrmVector("host_jad_block: "+path_def_pos[i], y1_host, y2);
			CPPUNIT_ASSERT( errRel < 1.0e-5 );

		}
	}


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


CPPUNIT_TEST_SUITE_REGISTRATION( JadBlockTestCase );

int main(int argc, char** argv)
{

	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest( registry.makeTest() );
	runner.run();
	return 0;

}



