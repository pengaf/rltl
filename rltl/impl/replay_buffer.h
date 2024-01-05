#pragma once
#include "utility.h"
#include "array.h"
#include "random.h"
#include "neural_network.h"
#include <tuple>

BEGIN_RLTL_IMPL

template<typename... T>
class ReplayBufferImpl;

template<>
class ReplayBufferImpl<>
{
public:
	struct AccessorTuple
	{};
public:
	void set(size_t index)
	{}
	void get(size_t dstIndex, size_t srcIndex, AccessorTuple& accessorTuple)
	{}
public:
	static void GetAccessorTuple(AccessorTuple& accessorTuple)
	{}
};

template<typename Value_t, typename... Rest_t>
class ReplayBufferImpl<Value_t, Rest_t...> : public ReplayBufferImpl<Rest_t...>
{
public:
	typedef Value_t Value_t;
	typedef ReplayBufferImpl<Rest_t...> Base_t;
	typedef Array_ElementType<Value_t> Element_t;
	typedef torch::TensorAccessor<Element_t, Array_Dimension<Value_t>::dim() + 1> Accessor_t;
	struct AccessorTuple : Base_t::AccessorTuple
	{
		Accessor_t accessor;
	};
public:
	void set(size_t index, const Value_t& element, const Rest_t&... rest)
	{
		m_vals[index] = element;
		Base_t::set(index, rest...);
	}
	void get(size_t dstIndex, size_t srcIndex, AccessorTuple& accessorTuple)
	{
		Tensor_Assign(accessorTuple.accessor[dstIndex], m_vals[srcIndex]);
		Base_t::get(dstIndex, srcIndex, accessorTuple);
	}
public:
	Value_t* m_vals;
public:
	template<typename Tensor_t, typename... OtherTensor_t>
	static void GetAccessorTuple(AccessorTuple& accessorTuple, Tensor_t& tensor, OtherTensor_t&... otherTensors)
	{
		accessorTuple.accessor = tensor.accessor<Element_t, Array_Dimension<Value_t>::dim() + 1>();
		Base_t::GetAccessorTuple(accessorTuple, otherTensors...)
	}
};

template<typename Value_t, typename... Rest_t>
class ReplayBuffer
{
public:
	typedef ReplayBufferImpl<Value_t, Rest_t...> ReplayBufferImpl_t;
public:
	ReplayBuffer(size_t capacity) :
		m_index(0),
		m_size(0)
	{
		m_capacity = capacity;
	}
public:
	void append(const Value_t& element, const Rest_t&... rest)
	{
		size_t index = m_index;
		m_impl.set(index, element, rest...);
		m_index = (index + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
	}
	template<typename Tensor_t, typename... Other_t>
	void sample(size_t batchSize, Tensor_t& tensor, Other_t&... others) const
	{
		assert(0 < batchSize && 0 < m_size);
		typename ReplayBufferImpl_t::AccessorTuple accessorTuple;
		ReplayBufferImpl_t::GetAccessorTuple(accessorTuple, tensor, otherTensors);
		for (size_t i = 0; i < batchSize; ++i)
		{
			size_t index = (size_t)Random::randint(m_size);
			assert(index < m_size);
			m_impl.get(i, index, accessorTuple);
		}
	}
protected:
	ReplayBufferImpl m_impl;
	size_t m_capacity{};
	size_t m_index{};
	size_t m_size{};
};


END_RLTL_IMPL