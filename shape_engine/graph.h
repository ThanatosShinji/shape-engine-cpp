#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>

#include "common.h"

namespace onnx_tool
{
	class Attribute
	{
	public:
		void deserialize(char*& buf)
		{
			int nlen = 0;
			int ntype = 0;
			Deserial(buf, &nlen);
			Deserial(buf, &ntype);
			for (int iele = 0; iele < nlen; iele++)
			{
				if (ntype == DTYPE_INT)
				{
					int tmp = 0;
					Deserial(buf, &tmp);
					mInts.push_back(tmp);
				}
				if (ntype == DTYPE_FLOAT)
				{
					float tmp = 0;
					Deserial(buf, &tmp);
					mFloats.push_back(tmp);
				}
				if (ntype == DTYPE_STR)
				{
					const char* ptr = NULL;
					Deserial(buf, &ptr);
					mStrings.push_back(ptr);
				}
			}
		}
		std::vector<int> mInts;
		std::vector<float> mFloats;
		std::vector<std::string> mStrings;
	};

	class Tensor
	{
	public:
		DType mDType;
		std::string mName;
		std::vector<int> mShape;
		uint64_t mSize;
		char* mRawptr;
	};

	class Node
	{
	public:
		std::string mName, mOpType;
		std::vector<std::string> mInputNames, mOutputNames;
		std::unordered_map<std::string, Attribute> mAttributes;
	};

	class Graph
	{
	public:
		Graph(FILE* _fp)
		{
			fseek(_fp, 0, SEEK_END);
			auto filelength = ftell(_fp);
			mRawBuf.resize(filelength);
			fseek(_fp, 0, SEEK_SET);
			fread(mRawBuf.data(), 1, filelength, _fp);
			deserialAll();
		}

		std::vector<Node> mNodes;
		std::vector<Tensor> mTensors;
		std::unordered_map<std::string, Tensor> mTensorMap;
	protected:
		void deserialAll()
		{
			char* ptr = mRawBuf.data();
			deserialNodes(ptr);
			deserialTensors(ptr);
		}

		void deserialNodes(char*& buf)
		{
			int n = 0;
			const char* ptr = NULL;
			Deserial(buf, &n);
			mNodes.resize(n);
			for (int i = 0; i < n; i++)
			{
				Deserial(buf, &ptr);
				mNodes[i].mName = ptr;
				Deserial(buf, &ptr);
				mNodes[i].mOpType = ptr;
				int ninput = 0;
				Deserial(buf, &ninput);
				for (int iin = 0; iin < ninput; iin++)
				{
					Deserial(buf, &ptr);
					mNodes[i].mInputNames.push_back(ptr);
				}
				int noutput = 0;
				Deserial(buf, &noutput);
				for (int iout = 0; iout < noutput; iout++)
				{
					Deserial(buf, &ptr);
					mNodes[i].mOutputNames.push_back(ptr);
				}
				int nattr = 0;
				Deserial(buf, &nattr);
				for (int iatt = 0; iatt < nattr; iatt++)
				{
					Deserial(buf, &ptr);
					Attribute attr;
					attr.deserialize(buf);
					mNodes[i].mAttributes[ptr] = attr;
				}
			}
		}

		void deserialTensors(char*& buf)
		{
			int num = 0;
			Deserial(buf, &num);
			mTensors.resize(num);
			for (int i = 0; i < num; i++)
			{
				const char* ptr = NULL;
				Deserial(buf, &ptr);
				mTensors[i].mName = ptr;
				int dtype = 0;
				Deserial(buf, &dtype);
				mTensors[i].mDType = static_cast<DType>(dtype);
				int shapelen = 0;
				Deserial(buf, &shapelen);
				mTensors[i].mShape.resize(shapelen);
				mTensors[i].mSize = GetEleSize(mTensors[i].mDType);
				for (int ish = 0; ish < shapelen; ish++)
				{
					int tmp = 0;
					Deserial(buf, &tmp);
					mTensors[i].mShape[ish] = tmp;
					mTensors[i].mSize *= tmp;
				}
				mTensors[i].mRawptr = buf;
				mTensorMap[mTensors[i].mName] = mTensors[i];
				buf += mTensors[i].mSize;
			}
		}

	public:
		static Graph* deserializeFile(const char* _filepath)
		{
			FILE* fp = fopen(_filepath, "rb");
			if (fp == NULL)
			{
				return NULL;
			}
			auto ptr = new Graph(fp);
			fclose(fp);
			return ptr;
		}

	private:
		std::vector<char> mRawBuf;

	};
}

