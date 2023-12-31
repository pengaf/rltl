#pragma once
#include "utility.h"
#include <torch/torch.h>
#include "../impl/array.h"

BEGIN_RLTL_NN

typedef torch::Tensor Tensor;

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t... t_sizes>
struct ArrayConstViewToTensorAccessor
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayConstView<Element_t, t_size_0, t_sizes...> ArrayConstView_t;
	ArrayConstViewToTensorAccessor(TensorAccessor_t& tensorAccessor, ArrayConstView_t arrayView)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			tensorAccessor[i] = arrayView[i];
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
struct ArrayConstViewToTensorAccessor<T, N, Element_t, t_size_0, t_size_1, t_sizes...>
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayConstView<Element_t, t_size_0, t_size_1, t_sizes...> ArrayConstView_t;
	ArrayConstViewToTensorAccessor(TensorAccessor_t& tensorAccessor, ArrayConstView_t arrayView)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			ArrayConstViewToTensorAccessor<T, N - 1, Element_t, t_size_1, t_sizes...>(tensorAccessor[i], arrayView[i]);
		}
	}
};


template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t... t_sizes>
struct ArrayViewToTensorAccessor
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayView<Element_t, t_size_0, t_sizes...> ArrayView_t;
	ArrayViewToTensorAccessor(TensorAccessor_t& tensorAccessor, ArrayView_t arrayView)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			tensorAccessor[i] = arrayView[i];
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
struct ArrayViewToTensorAccessor<T, N, Element_t, t_size_0, t_size_1, t_sizes...>
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayView<Element_t, t_size_0, t_size_1, t_sizes...> ArrayView_t;
	ArrayViewToTensorAccessor(TensorAccessor_t& tensorAccessor, ArrayView_t arrayView)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			ArrayViewToTensorAccessor<T, N - 1, Element_t, t_size_1, t_sizes...>(tensorAccessor[i], arrayView[i]);
		}
	}
};


template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t... t_sizes>
struct ArrayToTensorAccessor
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef Array<Element_t, t_size_0, t_sizes...> Array_t;
	ArrayToTensorAccessor(TensorAccessor_t& tensorAccessor, const Array_t& array)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			tensorAccessor[i] = array[i];
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
struct ArrayToTensorAccessor<T, N, Element_t, t_size_0, t_size_1, t_sizes...>
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef Array<Element_t, t_size_0, t_size_1, t_sizes...> Array_t;
	ArrayToTensorAccessor(TensorAccessor_t& tensorAccessor, const Array_t& array)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			ArrayConstViewToTensorAccessor(tensorAccessor[i], array[i]);
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assignTensor(torch::TensorAccessor<T, N>& tensorAccessor, ArrayConstView<Element_t, t_sizes...> arrayView)
{
	ArrayConstViewToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, arrayView);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assignTensor(torch::TensorAccessor<T, N>& tensorAccessor, ArrayView<Element_t, t_sizes...> arrayView)
{
	ArrayViewToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, arrayView);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assignTensor(torch::TensorAccessor<T, N>& tensorAccessor, const Array<Element_t, t_sizes...>& array)
{
	ArrayToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, array);
}

template<typename TensorAccessor_t, typename Element_t>
inline void assignTensor(TensorAccessor_t& tensorAccessor, Element_t value)
{
	assert(tensorAccessor.size(0) == 1);
	tensorAccessor[0] = value;
}


template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t... t_sizes>
struct TensorAccessorToArrayView
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayView<Element_t, t_size_0, t_sizes...> ArrayView_t;
	TensorAccessorToArrayView(ArrayView_t arrayView, const TensorAccessor_t& tensorAccessor)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			arrayView[i] = tensorAccessor[i];
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
struct TensorAccessorToArrayView<T, N, Element_t, t_size_0, t_size_1, t_sizes...>
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef ArrayView<Element_t, t_size_0, t_size_1, t_sizes...> ArrayView_t;
	TensorAccessorToArrayView(ArrayView_t arrayView, const TensorAccessor_t& tensorAccessor)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			TensorAccessorToArrayView<T, N - 1, Element_t, t_size_1, t_sizes...>(arrayView[i], tensorAccessor[i]);
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t... t_sizes>
struct TensorAccessorToArray
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef Array<Element_t, t_size_0, t_sizes...> Array_t;
	TensorAccessorToArray(Array_t& array, const TensorAccessor_t& tensorAccessor)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			array[i] = tensorAccessor[i];
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
struct TensorAccessorToArray<T, N, Element_t, t_size_0, t_size_1, t_sizes...>
{
	typedef torch::TensorAccessor<T, N> TensorAccessor_t;
	typedef Array<Element_t, t_size_0, t_size_1, t_sizes...> Array_t;
	TensorAccessorToArray(Array_t& array, const TensorAccessor_t& tensorAccessor)
	{
		assert(tensorAccessor.size(0) == t_size_0);
		for (size_t i = 0; i < t_size_0; ++i)
		{
			TensorAccessorToArrayView(array[i], tensorAccessor[i]);
		}
	}
};

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void retrieveTensor(ArrayView<Element_t, t_sizes...> arrayView, const torch::TensorAccessor<T, N>& tensorAccessor)
{
	TensorAccessorToArrayView<T, N, Element_t, t_sizes...>(arrayView, tensorAccessor);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void retrieveTensor(Array<Element_t, t_sizes...>& array, const torch::TensorAccessor<T, N>& tensorAccessor)
{
	TensorAccessorToArray<T, N, Element_t, t_sizes...>(array, tensorAccessor);
}

template<typename TensorAccessor_t, typename Element_t>
inline void retrieveTensor(Element_t& value, const TensorAccessor_t& tensorAccessor)
{
	assert(tensorAccessor.size(0) == 1);
	value = tensorAccessor[0];
}

END_RLTL_NN