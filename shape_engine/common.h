#pragma once
#include <stdint.h>
#include <string>

namespace onnx_tool
{
	enum DType
	{
		DTYPE_INT = 0,
		DTYPE_FLOAT = 1,
		DTYPE_STR = 2,
		DTYPE_FP16 = 3,
		DTYPE_DOUBLE = 4,
		DTYPE_INT8 = 5,
		DTYPE_UINT8 = 6,
		DTYPE_INT16 = 7,
		DTYPE_UINT16 = 8,
		DTYPE_INT64 = 9,
	};

	template<typename T>
	inline void Deserial(char*& buf, T* val)
	{
		auto ptr = (T*)buf;
		*val = *ptr;
		buf += sizeof(T);
	}

	template<>
	inline void Deserial(char*& buf, const char** val)
	{
		auto ptr = (const char*)buf;
		*val = ptr;
		buf = buf + strlen(ptr) + 1;
	}

	template<>
	inline void Deserial(char*& buf, char** val)
	{
		auto ptr = (char*)buf;
		*val = ptr;
		buf = buf + strlen(ptr) + 1;
	}

	inline int GetEleSize(DType _dtype)
	{
		switch (_dtype)
		{
		case DTYPE_STR:
			return -1;
		case DTYPE_DOUBLE:
		case DTYPE_INT64:
			return 8;
		case DTYPE_INT8:
		case DTYPE_UINT8:
			return 1;
		case DTYPE_FP16:
		case DTYPE_INT16:
		case DTYPE_UINT16:
			return 2;
		case DTYPE_INT:
		case DTYPE_FLOAT:
		default:
			return 4;
		}
	}
}

