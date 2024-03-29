﻿#pragma once
#include "common.h"
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx_tool {
namespace shape_engine {
struct TensorShape {
  int n;
  int *ptr;
};

static inline size_t shape_size(TensorShape &_shape) {
  size_t _s = 1;
  _s = std::accumulate(_shape.ptr, _shape.ptr + _shape.n, _s,
                       std::multiplies<int>());
  return _s;
}

class ValueExpr {
public:
  ValueExpr() : mSrcVariable("") {
    mAlpha = 1;
    mBeta = 1;
    mFactor = 0;
    mTrunc = 0;
  }

  void deserial(char *&buf) {
    const char *ptr;
    Deserial(buf, &ptr);
    mSrcVariable = ptr;
    Deserial(buf, &mAlpha);
    Deserial(buf, &mBeta);
    Deserial(buf, &mFactor);
    Deserial(buf, &mTrunc);
  }

  int operator()(int x) {
    float y = 0.f;
    if (mFactor == 0) {
      y = mAlpha * x + mBeta;
    } else {
      y = x / mAlpha + mBeta;
    }
    if (mTrunc == 0) {
      return std::ceil(y);
    } else {
      return std::floor(y);
    }
  }
  std::string mSrcVariable, mTarVariable;

private:
  float mAlpha, mBeta;
  int mFactor, mTrunc;
};

class TensorDesc {
public:
  void addNumber(int _num) { mShape.push_back(_num); }

  void addVariable(std::string _var) {
    mIndice.push_back(mShape.size());
    mVariables.push_back(_var);
    mShape.push_back(0);
  }

  std::vector<int> mIndice, mShape;
  std::vector<std::string> mVariables;
};

class ShapeEngine {
public:
  ShapeEngine() {}

  virtual void update_variable(const char *_var_name, int _val) {
    mVariables[_var_name] = _val;
  }

  virtual void update_variables() {
    for (int i = 0; i < mValueExpr.size(); i++) {
      auto &expr = mValueExpr[i];
      mVariables[expr.mTarVariable] = expr(mVariables[expr.mSrcVariable]);
    }
    for (auto [key, val] : mTensorDesc) {
      auto &desc = mTensorDesc[key];
      for (int i = 0; i < desc.mIndice.size(); i++) {
        desc.mShape[desc.mIndice[i]] = mVariables[desc.mVariables[i]];
      }
    }
  }

  virtual int get_tensor_shape_len(const char *_tname) {
    return mTensorDesc[_tname].mShape.size();
  }

  virtual int *get_tensor_shape_ptr(const char *_tname) {
    return mTensorDesc[_tname].mShape.data();
  }

  std::vector<std::string> mDynamicTensors;

  void deserializeFile(const char *_filepath) {
    FILE *fp = fopen(_filepath, "rb");
    if (fp == NULL) {
      return;
    }
    fseek(fp, 0, SEEK_END);
    auto filelength = ftell(fp);
    mRawBuf.resize(filelength);
    fseek(fp, 0, SEEK_SET);
    fread(mRawBuf.data(), 1, filelength, fp);
    deserialAll();
    fclose(fp);
  }

protected:
  void deserialAll() {
    char *ptr = mRawBuf.data();
    deserialVariableNames(ptr);
    deserialTensorDesc(ptr);
    deserialTensorExpr(ptr);
  }

  void deserialVariableNames(char *&buf) {
    const char *ptr = NULL;
    int num = 0;
    Deserial(buf, &num);
    for (int i = 0; i < num; i++) {
      Deserial(buf, &ptr);
      mVariables[std::string(ptr)] = 0;
    }
  }

  void deserialTensorDesc(char *&buf) {
    const char *ptr = NULL;
    int num = 0;
    Deserial(buf, &num);
    for (int i = 0; i < num; i++) {
      Deserial(buf, &ptr);
      auto tname = std::string(ptr);
      int dnum = 0;
      Deserial(buf, &dnum);
      TensorDesc td;
      for (int j = 0; j < dnum; j++) {
        int dtype = 0;
        Deserial(buf, &dtype);
        if (dtype == DTYPE_INT) {
          int val = 0;
          Deserial(buf, &val);
          td.addNumber(val);
        }
        if (dtype == DTYPE_STR) {
          Deserial(buf, &ptr);
          td.addVariable(ptr);
        }
      }
      mDynamicTensors.push_back(tname);
      mTensorDesc[tname] = td;
    }
  }

  void deserialTensorExpr(char *&buf) {
    const char *ptr = NULL;
    int num = 0;
    Deserial(buf, &num);
    for (int i = 0; i < num; i++) {
      Deserial(buf, &ptr);
      ValueExpr expr;
      expr.mTarVariable = ptr;
      expr.deserial(buf);
      mValueExpr.push_back(expr);
    }
  }
  std::vector<char> mRawBuf;
  std::unordered_map<std::string, int> mVariables;
  std::vector<ValueExpr> mValueExpr;
  std::unordered_map<std::string, TensorDesc> mTensorDesc;
};

} // namespace shape_engine

class MemoryMap {
public:
  typedef std::array<int64_t, 2> size_block_t;
  MemoryMap() : mMaxSize(0) {}

  virtual int64_t get_total_size() const { return mMaxSize; }

  virtual int64_t get_tensor_offset(const char *_tname) {
    return mTensorBlock[_tname][0];
  }

  virtual int64_t get_tensor_size(const char *_tname) {
    return mTensorBlock[_tname][1];
  }

  virtual std::vector<std::pair<std::string, int>> &get_max_shape() {
    return mMaxInputShape;
  }

  void deserializeFile(const char *_filepath) {
    FILE *fp = fopen(_filepath, "rb");
    if (fp == NULL) {
      return;
    }
    fseek(fp, 0, SEEK_END);
    auto filelength = ftell(fp);
    mRawBuf.resize(filelength);
    fseek(fp, 0, SEEK_SET);
    fread(mRawBuf.data(), 1, filelength, fp);
    deserialAll();
    fclose(fp);
  }

  void deserializeBuffer(char *buf, size_t size) {
    mRawBuf.resize(size);
    std::memcpy(mRawBuf.data(), buf, size);
    deserialAll();
  }

protected:
  void deserialAll() {
    char *ptr = mRawBuf.data();
    Deserial(ptr, &mMaxSize);
    deserialTensorBlock(ptr);
    deseialMaxShape(ptr);
  }
  void deserialTensorBlock(char *&buf) {
    const char *ptr = NULL;
    int64_t num = 0;
    Deserial(buf, &num);
    for (int64_t i = 0; i < num; i++) {
      Deserial(buf, &ptr);
      int64_t tmp0, tmp1;
      Deserial(buf, &tmp0);
      Deserial(buf, &tmp1);
      mTensorBlock[ptr] = size_block_t{tmp0, tmp1};
    }
  }
  void deseialMaxShape(char *&buf) {
    const char *ptr = NULL;
    int64_t num = 0;
    Deserial(buf, &num);
    for (int64_t i = 0; i < num; i++) {
      Deserial(buf, &ptr);
      int64_t tmp0;
      Deserial(buf, &tmp0);
      mMaxInputShape.push_back({ptr, int(tmp0)});
    }
  }
  std::vector<char> mRawBuf;
  int64_t mMaxSize;
  std::unordered_map<std::string, size_block_t> mTensorBlock;
  std::vector<std::pair<std::string, int>> mMaxInputShape;
};

} // namespace onnx_tool
