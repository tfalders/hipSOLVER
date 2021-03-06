/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrf.hpp"
#include "testing_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda}

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100},
};

const vector<int> n_size_range = {
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    40,
    100,
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {
    45,
    64,
    520,
    1024,
    2000,
};

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int         n_size      = std::get<1>(tup);

    Arguments arg;

    arg.M   = matrix_size[0];
    arg.N   = n_size;
    arg.lda = matrix_size[1];

    arg.timing   = 0;
    arg.singular = matrix_size[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = min(arg.M, arg.N);
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GETRF : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETRF_FORTRAN : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF_FORTRAN() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETRF_NPVT : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETRF_NPVT_FORTRAN : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF_NPVT_FORTRAN() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests
TEST_P(GETRF, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<false, false, false, float>(arg);
}

TEST_P(GETRF, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<false, false, false, double>(arg);
}

TEST_P(GETRF, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<false, false, false, hipsolverComplex>(arg);
}

TEST_P(GETRF, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<false, false, false, hipsolverDoubleComplex>(arg);
}

TEST_P(GETRF_FORTRAN, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<true, false, false, float>(arg);
}

TEST_P(GETRF_FORTRAN, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<true, false, false, double>(arg);
}

TEST_P(GETRF_FORTRAN, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<true, false, false, hipsolverComplex>(arg);
}

TEST_P(GETRF_FORTRAN, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf<true, false, false, hipsolverDoubleComplex>(arg);
}
TEST_P(GETRF_NPVT, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<false, false, false, float>(arg);
}

TEST_P(GETRF_NPVT, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<false, false, false, double>(arg);
}

TEST_P(GETRF_NPVT, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<false, false, false, hipsolverComplex>(arg);
}

TEST_P(GETRF_NPVT, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<false, false, false, hipsolverDoubleComplex>(arg);
}

TEST_P(GETRF_NPVT_FORTRAN, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<true, false, false, float>(arg);
}

TEST_P(GETRF_NPVT_FORTRAN, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<true, false, false, double>(arg);
}

TEST_P(GETRF_NPVT_FORTRAN, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<true, false, false, hipsolverComplex>(arg);
}

TEST_P(GETRF_NPVT_FORTRAN, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    arg.batch_count = 1;
    testing_getrf_npvt<true, false, false, hipsolverDoubleComplex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
