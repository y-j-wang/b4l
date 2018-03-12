#ifndef BASEUTIL_H
#define BASEUTIL_H

#include <iostream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <list>
#include <functional>
#include <thread>
#include <string>
#include <mkl.h>
#include <mkl_vml.h>

using std::cout;
using std::cerr;
using std::endl;

using std::string;
using std::vector;
using std::set;
using std::list;
using std::make_pair;
using std::pair;
using std::map;
using std::unordered_map;
using std::unordered_set;
using std::to_string;

////////////////// Intel Threading Building Blocks //////////////////////
#include <tbb/tbb.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>

using namespace tbb;

////////////////// Intel Threading Building Blocks //////////////////////

////////////////// Boost //////////////////////

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;

////////////////// Boost //////////////////////

////////////////// Types //////////////////////
typedef unsigned long long ull;

#define use_double
#define use_mkl
#define detailed_eval

#ifdef use_double
typedef double value_type;
#define LAPACKE_xsyevr LAPACKE_dsyevr
#define LAPACKE_xgesvd LAPACKE_dgesvd
#define LAPACKE_xgesv LAPACKE_dgesv
#define LAPACKE_xlamch LAPACKE_dlamch
#define cblas_xgemm cblas_dgemm
#define vxMul vdMul
#define vxAdd vdAdd
#define vxDiv vdDiv
#define vxSub vdSub
#define cblas_xnrm2 cblas_dnrm2
#define mkl_xcsrmm mkl_dcsrmm
#define mkl_ximatcopy mkl_dimatcopy
#define cblas_xdot cblas_ddot
#define vxSqrt vdSqrt
#define mkl_xomatadd mkl_domatadd
#define cblas_ixamax cblas_idamax
#define mkl_ximatcopy mkl_dimatcopy
#define mkl_xomatcopy mkl_domatcopy
#define cblas_xaxpy cblas_daxpy
#define cblas_xscal cblas_dscal
#define cblas_xcopy cblas_dcopy
#else
typedef float value_type;
    #define LAPACKE_xsyevr LAPACKE_ssyevr
    #define LAPACKE_xgesvd LAPACKE_sgesvd
    #define LAPACKE_xgesv LAPACKE_sgesv
    #define LAPACKE_xlamch LAPACKE_slamch
    #define cblas_xgemm cblas_sgemm
    #define vxMul vsMul
    #define vxAdd vsAdd
    #define vxDiv vsDiv
    #define vxSub vsSub
    #define cblas_xnrm2 cblas_snrm2
    #define mkl_xcsrmm mkl_scsrmm
    #define mkl_ximatcopy mkl_simatcopy
    #define cblas_xdot cblas_sdot
    #define vxSqrt vsSqrt
    #define mkl_xomatadd mkl_somatadd
    #define cblas_ixamax cblas_isamax
    #define mkl_ximatcopy mkl_simatcopy
    #define mkl_xomatcopy mkl_somatcopy
    #define cblas_xaxpy cblas_saxpy
    #define cblas_xscal cblas_sscal
    #define cblas_xcopy cblas_scopy
#endif

namespace mf {
    // common matrix types
    typedef boost::numeric::ublas::compressed_matrix<value_type, boost::numeric::ublas::row_major>
            SparseMatrix;
    typedef boost::numeric::ublas::coordinate_matrix<value_type, boost::numeric::ublas::column_major>
            SparseMatrixCM;
    typedef boost::numeric::ublas::matrix<value_type, boost::numeric::ublas::row_major>
            DenseMatrix;
    typedef boost::numeric::ublas::matrix<value_type, boost::numeric::ublas::column_major>
            DenseMatrixCM;
    typedef boost::numeric::ublas::vector<value_type>
            Vec;
    typedef SparseMatrix::size_type mf_size_type;
}

using namespace mf;

enum Method{m_RESCAL=0, m_RESCAL_RANK=1, m_TransE=2, m_HOLE=3, m_RTLREnsemble=4, m_HTLREnsemble=5, m_RHLREnsemble=6, m_RHTLREnsemble=7, m_THPipeline=8, m_Ensemble=9};

////////////////// Types //////////////////////


#endif //BASEUTIL_H
