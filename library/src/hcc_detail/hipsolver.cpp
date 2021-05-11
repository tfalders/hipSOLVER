/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver.h"
#include "exceptions.hpp"
#include "rocblas.h"
#include "rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>
#include <mutex>
#include <unordered_map>

using namespace std;

/******************** WORKSPACE ********************/
struct workspace_info
{
    void*  work;
    size_t lwork;

    // Constructor
    explicit workspace_info() = default;

    workspace_info(void* ptr, size_t size)
        : work(ptr)
        , lwork(size)
    {
    }
};

mutex                                         w_mutex;
unordered_map<rocblas_handle, workspace_info> workspaces;

rocblas_status hipsolverSetWorkspace(rocblas_handle handle, void* work, size_t lwork)
{
    workspace_info* info;
    {
        const lock_guard<mutex> lock(w_mutex);
        info = &workspaces[handle];
    }

    // if workspace is unchanged from last call, do nothing
    if(work == info->work && lwork == info->lwork)
        return rocblas_status_success;

    // complete all calculations before changing the workspace
    if(info->lwork > 0)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);
        hipStreamSynchronize(stream);
    }

    // change the workspace
    rocblas_status status;
    if(work != nullptr)
        status = rocblas_set_workspace(handle, work, lwork);
    else
    {
        if(info->work != nullptr || info->lwork < lwork)
            status = rocblas_set_device_memory_size(handle, lwork);
    }

    // update the map
    if(status == rocblas_status_success)
    {
        info->work = work;
        rocblas_get_device_memory_size(handle, &info->lwork);
    }
    return status;
}

#define HIPSOLVER_CALL_FUNC(handle, work, lwork, func_call)                                   \
    try                                                                                       \
    {                                                                                         \
        size_t _lwork = lwork;                                                                \
        if(work == nullptr || lwork == 0)                                                     \
        {                                                                                     \
            rocblas_start_device_memory_size_query((rocblas_handle)handle);                   \
            func_call;                                                                        \
            rocblas_stop_device_memory_size_query((rocblas_handle)handle, &_lwork);           \
        }                                                                                     \
                                                                                              \
        rocblas_status _status = hipsolverSetWorkspace((rocblas_handle)handle, work, _lwork); \
        if(_status != rocblas_status_success)                                                 \
            return rocblas2hip_status(_status);                                               \
                                                                                              \
        return rocblas2hip_status(func_call);                                                 \
    }                                                                                         \
    catch(...)                                                                                \
    {                                                                                         \
        return exception2hip_status();                                                        \
    }

extern "C" {

/******************** CONVERSIONS ********************/
rocblas_operation_ hip2rocblas_operation(hipsolverOperation_t op)
{
    switch(op)
    {
    case HIPSOLVER_OP_N:
        return rocblas_operation_none;
    case HIPSOLVER_OP_T:
        return rocblas_operation_transpose;
    case HIPSOLVER_OP_C:
        return rocblas_operation_conjugate_transpose;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverOperation_t rocblas2hip_operation(rocblas_operation_ op)
{
    switch(op)
    {
    case rocblas_operation_none:
        return HIPSOLVER_OP_N;
    case rocblas_operation_transpose:
        return HIPSOLVER_OP_T;
    case rocblas_operation_conjugate_transpose:
        return HIPSOLVER_OP_C;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

rocblas_fill_ hip2rocblas_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return rocblas_fill_upper;
    case HIPSOLVER_FILL_MODE_LOWER:
        return rocblas_fill_lower;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverFillMode_t rocblas2hip_fill(rocblas_fill_ fill)
{
    switch(fill)
    {
    case rocblas_fill_upper:
        return HIPSOLVER_FILL_MODE_UPPER;
    case rocblas_fill_lower:
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverStatus_t rocblas2hip_status(rocblas_status_ error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPSOLVER_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case rocblas_status_invalid_size:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    default:
        return HIPSOLVER_STATUS_UNKNOWN;
    }
}

/******************** AUXILIARY ********************/
hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;

    // Create the rocBLAS handle
    rocblas_status status = rocblas_create_handle((rocblas_handle*)handle);
    if(status == rocblas_status_success)
    {
        const lock_guard<mutex> lock(w_mutex);
        workspaces[(rocblas_handle)handle] = workspace_info(nullptr, 0);
    }

    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
try
{
    rocblas_status status = rocblas_destroy_handle((rocblas_handle)handle);
    if(status == rocblas_status_success)
    {
        const lock_guard<mutex> lock(w_mutex);
        workspaces.erase((rocblas_handle)handle);
    }

    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return rocblas2hip_status(rocblas_set_stream((rocblas_handle)handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return rocblas2hip_status(rocblas_get_stream((rocblas_handle)handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GEQRF ********************/
hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_sgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_dgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_cgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_zgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            tau,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo)
{
    HIPSOLVER_CALL_FUNC(
        handle, work, lwork, rocsolver_sgeqrf((rocblas_handle)handle, m, n, A, lda, tau));
}

hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           tau,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo)
{
    HIPSOLVER_CALL_FUNC(
        handle, work, lwork, rocsolver_dgeqrf((rocblas_handle)handle, m, n, A, lda, tau));
}

hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* tau,
                                  hipsolverComplex* work,
                                  int               lwork,
                                  int*              devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        work,
                        lwork,
                        rocsolver_cgeqrf((rocblas_handle)handle,
                                         m,
                                         n,
                                         (rocblas_float_complex*)A,
                                         lda,
                                         (rocblas_float_complex*)tau));
}

hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* tau,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        work,
                        lwork,
                        rocsolver_zgeqrf((rocblas_handle)handle,
                                         m,
                                         n,
                                         (rocblas_double_complex*)A,
                                         lda,
                                         (rocblas_double_complex*)tau));
}

/******************** GETRF ********************/
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_sgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    if(max(sz1, sz2) > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_dgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    if(max(sz1, sz2) > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
try
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_cgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_cgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    if(max(sz1, sz2) > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
try
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_zgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_zgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    if(max(sz1, sz2) > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_sgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    }
    else
    {
        HIPSOLVER_CALL_FUNC(
            handle, work, 0, rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
    }
}

hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_dgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    }
    else
    {
        HIPSOLVER_CALL_FUNC(
            handle, work, 0, rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
    }
}

hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_cgetrf(
                (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devIpiv, devInfo));
    }
    else
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_cgetrf_npvt(
                (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devInfo));
    }
}

hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* work,
                                  int*                    devIpiv,
                                  int*                    devInfo)
{
    if(devIpiv != nullptr)
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_zgetrf(
                (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devIpiv, devInfo));
    }
    else
    {
        HIPSOLVER_CALL_FUNC(
            handle,
            work,
            0,
            rocsolver_zgetrf_npvt(
                (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devInfo));
    }
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  float*               A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  float*               B,
                                  int                  ldb,
                                  int*                 devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_sgetrs((rocblas_handle)handle,
                                         hip2rocblas_operation(trans),
                                         n,
                                         nrhs,
                                         A,
                                         lda,
                                         devIpiv,
                                         B,
                                         ldb));
}

hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  double*              A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  double*              B,
                                  int                  ldb,
                                  int*                 devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_dgetrs((rocblas_handle)handle,
                                         hip2rocblas_operation(trans),
                                         n,
                                         nrhs,
                                         A,
                                         lda,
                                         devIpiv,
                                         B,
                                         ldb));
}

hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipsolverComplex*    A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipsolverComplex*    B,
                                  int                  ldb,
                                  int*                 devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_cgetrs((rocblas_handle)handle,
                                         hip2rocblas_operation(trans),
                                         n,
                                         nrhs,
                                         (rocblas_float_complex*)A,
                                         lda,
                                         devIpiv,
                                         (rocblas_float_complex*)B,
                                         ldb));
}

hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t       handle,
                                  hipsolverOperation_t    trans,
                                  int                     n,
                                  int                     nrhs,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  int*                    devIpiv,
                                  hipsolverDoubleComplex* B,
                                  int                     ldb,
                                  int*                    devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_zgetrs((rocblas_handle)handle,
                                         hip2rocblas_operation(trans),
                                         n,
                                         nrhs,
                                         (rocblas_double_complex*)A,
                                         lda,
                                         devIpiv,
                                         (rocblas_double_complex*)B,
                                         ldb));
}

/******************** POTRF ********************/
hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_spotrf(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_dpotrf(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipsolverComplex*   A,
                                             int                 lda,
                                             int*                lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_cpotrf(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t       handle,
                                             hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             int*                    lwork)
try
{
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status = rocsolver_zpotrf(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return rocblas2hip_status(status);
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
{
    HIPSOLVER_CALL_FUNC(
        handle,
        work,
        lwork,
        rocsolver_spotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
}

hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
{
    HIPSOLVER_CALL_FUNC(
        handle,
        work,
        lwork,
        rocsolver_dpotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
}

hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipsolverComplex*   A,
                                  int                 lda,
                                  hipsolverComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        work,
                        lwork,
                        rocsolver_cpotrf((rocblas_handle)handle,
                                         hip2rocblas_fill(uplo),
                                         n,
                                         (rocblas_float_complex*)A,
                                         lda,
                                         devInfo));
}

hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t       handle,
                                  hipsolverFillMode_t     uplo,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
{
    HIPSOLVER_CALL_FUNC(handle,
                        work,
                        lwork,
                        rocsolver_zpotrf((rocblas_handle)handle,
                                         hip2rocblas_fill(uplo),
                                         n,
                                         (rocblas_double_complex*)A,
                                         lda,
                                         devInfo));
}

/******************** POTRF_BATCHED ********************/
hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         int*                devInfo,
                                         int                 batch_count)
{
    HIPSOLVER_CALL_FUNC(
        handle,
        nullptr,
        0,
        rocsolver_spotrf_batched(
            (rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo, batch_count));
}

hipsolverStatus_t hipsolverDpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A[],
                                         int                 lda,
                                         int*                devInfo,
                                         int                 batch_count)
{
    HIPSOLVER_CALL_FUNC(
        handle,
        nullptr,
        0,
        rocsolver_dpotrf_batched(
            (rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo, batch_count));
}

hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int*                devInfo,
                                         int                 batch_count)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_cpotrf_batched((rocblas_handle)handle,
                                                 hip2rocblas_fill(uplo),
                                                 n,
                                                 (rocblas_float_complex**)A,
                                                 lda,
                                                 devInfo,
                                                 batch_count));
}

hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int*                    devInfo,
                                         int                     batch_count)
{
    HIPSOLVER_CALL_FUNC(handle,
                        nullptr,
                        0,
                        rocsolver_zpotrf_batched((rocblas_handle)handle,
                                                 hip2rocblas_fill(uplo),
                                                 n,
                                                 (rocblas_double_complex**)A,
                                                 lda,
                                                 devInfo,
                                                 batch_count));
}

} // extern C
