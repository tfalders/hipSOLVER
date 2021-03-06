/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "../include/hipsolver_datatype2string.hpp"

/* ============================================================================================ */
/*  Convert hipsolver constants to lapack char. */

char hipsolver2char_operation(hipsolverOperation_t value)
{
    switch(value)
    {
    case HIPSOLVER_OP_N:
        return 'N';
    case HIPSOLVER_OP_T:
        return 'T';
    case HIPSOLVER_OP_C:
        return 'C';
    }
    return '\0';
}

char hipsolver2char_fill(hipsolverFillMode_t value)
{
    switch(value)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return 'U';
    case HIPSOLVER_FILL_MODE_LOWER:
        return 'L';
    case HIPSOLVER_FILL_MODE_FULL:
        return 'F';
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to hipsolver type. */

hipsolverStatus_t string2hipsolver_status(const std::string& value)
{
    return value == "HIPSOLVER_STATUS_SUCCESS"             ? HIPSOLVER_STATUS_SUCCESS
           : value == "HIPSOLVER_STATUS_NOT_INITIALIZED"   ? HIPSOLVER_STATUS_NOT_INITIALIZED
           : value == "HIPSOLVER_STATUS_ALLOC_FAILED"      ? HIPSOLVER_STATUS_ALLOC_FAILED
           : value == "HIPSOLVER_STATUS_INVALID_VALUE"     ? HIPSOLVER_STATUS_INVALID_VALUE
           : value == "HIPSOLVER_STATUS_MAPPING_ERROR"     ? HIPSOLVER_STATUS_MAPPING_ERROR
           : value == "HIPSOLVER_STATUS_EXECUTION_FAILED"  ? HIPSOLVER_STATUS_EXECUTION_FAILED
           : value == "HIPSOLVER_STATUS_INTERNAL_ERROR"    ? HIPSOLVER_STATUS_INTERNAL_ERROR
           : value == "HIPSOLVER_STATUS_NOT_SUPPORTED"     ? HIPSOLVER_STATUS_NOT_SUPPORTED
           : value == "HIPSOLVER_STATUS_ARCH_MISMATCH"     ? HIPSOLVER_STATUS_ARCH_MISMATCH
           : value == "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR" ? HIPSOLVER_STATUS_HANDLE_IS_NULLPTR
                                                           : static_cast<hipsolverStatus_t>(-1);
}

hipsolverOperation_t char2hipsolver_operation(char value)
{
    switch(value)
    {
    case 'N':
        return HIPSOLVER_OP_N;
    case 'T':
        return HIPSOLVER_OP_T;
    case 'C':
        return HIPSOLVER_OP_C;
    case 'n':
        return HIPSOLVER_OP_N;
    case 't':
        return HIPSOLVER_OP_T;
    case 'c':
        return HIPSOLVER_OP_C;
    }
    return HIPSOLVER_OP_N;
}

hipsolverFillMode_t char2hipsolver_fill(char value)
{
    switch(value)
    {
    case 'U':
        return HIPSOLVER_FILL_MODE_UPPER;
    case 'L':
        return HIPSOLVER_FILL_MODE_LOWER;
    case 'u':
        return HIPSOLVER_FILL_MODE_UPPER;
    case 'l':
        return HIPSOLVER_FILL_MODE_LOWER;
    }
    return HIPSOLVER_FILL_MODE_LOWER;
}
