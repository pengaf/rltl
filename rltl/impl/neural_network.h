#pragma once
#include "utility.h"
#include "array.h"
#include "random.h"

BEGIN_RLTL_IMPL


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
inline void assign(torch::TensorAccessor<T, N>& tensorAccessor, ArrayConstView<Element_t, t_sizes...> arrayView)
{
	ArrayConstViewToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, arrayView);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assign(torch::TensorAccessor<T, N>& tensorAccessor, ArrayView<Element_t, t_sizes...> arrayView)
{
	ArrayViewToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, arrayView);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assign(torch::TensorAccessor<T, N>& tensorAccessor, const Array<Element_t, t_sizes...>& array)
{
	ArrayToTensorAccessor<T, N, Element_t, t_sizes...>(tensorAccessor, array);
}

template<typename T>//, typename Element_t>
inline void assign(torch::TensorAccessor<T, 1>& tensorAccessor, float value)
{
	assert(tensorAccessor.size(0) == 1);
	tensorAccessor[0] = value;
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assign(ArrayView<Element_t, t_sizes...> arrayView, const torch::TensorAccessor<T, N>& tensorAccessor)
{
	TensorAccessorToArrayView<T, N, Element_t, t_sizes...>(arrayView, tensorAccessor);
}

template<typename T, size_t N, typename Element_t, size_t... t_sizes>
inline void assign(Array<Element_t, t_sizes...>& array, const torch::TensorAccessor<T, N>& tensorAccessor)
{
	TensorAccessorToArray<T, N, Element_t, t_sizes...>(array, tensorAccessor);
}

template<typename T, typename Element_t>
inline void assign(Element_t& value, const torch::TensorAccessor<T, 1>& tensorAccessor)
{
	assert(tensorAccessor.size(0) == 1);
	value = tensorAccessor[0];
}

template<typename Element_t, typename TensorScalar_t>
Tensor NN_makeTensor(TensorScalar_t dtype, uint32_t batchSize)
{
	auto shape = GetShape<Element_t>::shape();
	std::array<int64_t, GetDimension<Element_t>::dim() + 1> tensorShape;
	tensorShape[0] = batchSize;
	for (size_t i = 0; i < GetDimension<Element_t>::dim(); ++i)
	{
		tensorShape[i + 1] = shape[i];
	}
	Tensor tensor = torch::empty(tensorShape, torch::TensorOptions().dtype(dtype));
	return tensor;
}

template<typename Network_t, typename State_t>
inline float NN_getStateValue(Network_t& network, const State_t& state)
{
	auto shape = GetShape<State_t>::shape();
	std::array<int64_t, GetDimension<State_t>::dim() + 1> tensorShape;
	tensorShape[0] = 1;
	for (size_t i = 0; i < State_t::t_dim; ++i)
	{
		tensorShape[i + 1] = shape[i];
	}
	torch::Tensor stateTensor = torch::empty(tensorShape, torch::TensorOptions().dtype(torch::kFloat32));
	auto stateAccessor = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	assign(stateAccessor[0], state);
	torch::Tensor valueTensor = network->forward(stateTensor);
	auto valueAccessor = valueTensor.accessor<int64_t, 2>();
	float value;
	assign(value, valueAccessor[0]);
	return value;
}

template<typename Network_t, typename State_t>
inline void NN_getStateValues(std::vector<float>& values, Network_t& network, const State_t& state)
{
	auto shape = GetShape<State_t>::shape();
	std::array<int64_t, GetDimension<State_t>::dim() + 1> tensorShape;
	tensorShape[0] = 1;
	for (size_t i = 0; i < State_t::t_dim; ++i)
	{
		tensorShape[i + 1] = shape[i];
	}
	torch::Tensor stateTensor = torch::empty(tensorShape, torch::TensorOptions().dtype(torch::kFloat32));
	auto stateAccessor = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	assign(stateAccessor[0], state);
	torch::Tensor valueTensor = network->forward(stateTensor);
	auto valueAccessor = valueTensor.accessor<int64_t, 2>();	
	size_t count = valueTensor.size(1);
	values.resize(count);
	for (uint32_t i = 0; i < m_actionCount; ++i)
	{
		values[i] = valueAccessor[0][i];
	}
}

template<typename Network_t, typename State_t, typename Action_t>
inline Action_t NN_actionByArgmax(Network_t& network, const State_t& state)
{
	assert(GetDimension<Action_t>::dim() == 1);
	auto shape = GetShape<State_t>::shape();
	std::array<int64_t, GetDimension<State_t>::dim() + 1> tensorShape;
	tensorShape[0] = 1;
	for (size_t i = 0; i < State_t::t_dim; ++i)
	{
		tensorShape[i + 1] = shape[i];
	}
	torch::Tensor stateTensor = torch::empty(tensorShape, torch::TensorOptions().dtype(torch::kFloat32));
	auto stateAccessor = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	assign(stateAccessor[0], state);
	torch::Tensor actionTensor = network->actionValue(stateTensor).argmax(1);
	auto actionAccessor = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim()>();
	Action_t action;
	assign(action, actionAccessor);
	return action;
}

template<typename Network_t, typename State_t, typename Action_t>
inline Action_t NN_actionBySoftmax(Network_t& network, const State_t& state)
{
	assert(GetDimension<Action_t>::dim() == 1);
	auto shape = GetShape<State_t>::shape();
	std::array<int64_t, GetDimension<State_t>::dim() + 1> tensorShape;
	tensorShape[0] = 1;
	for (size_t i = 0; i < State_t::t_dim; ++i)
	{
		tensorShape[i + 1] = shape[i];
	}
	torch::Tensor stateTensor = torch::empty(tensorShape, torch::TensorOptions().dtype(torch::kFloat32));
	auto stateAccessor = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	assign(stateAccessor[0], state);
	torch::Tensor probTensor = torch::nn::functional::softmax(network->logitAction(stateTensor), 1);
	size_t probSize = probTensor.size(1);

	Action_t action = probSize - 1;
	auto probAccessor = probTensor.accessor<float, 2>();
	float rnd = Random::rand();
	for (size_t i = 0; i < probSize; ++i)
	{
		float prob = probAccessor[0][i];
		if (rnd <= prob)
		{
			action = i;
			break;
		}
		rnd -= prob;
	}
	return action;
}


template<typename Net_t>
inline void NN_copyParameters(Net_t& dst, const Net_t& src)
{
	std::stringstream stream;
	torch::serialize::OutputArchive outputArchive(std::make_shared<torch::jit::CompilationUnit>());
	torch::serialize::InputArchive inputArchive;
	src->save(outputArchive);
	outputArchive.save_to(stream);
	inputArchive.load_from(stream);
	dst->load(inputArchive);
}

template<typename Net_t>
inline void NN_printParameters(Net_t& net)
{
	std::cout << net;
	auto nps = net->named_parameters();
	for (auto it = nps.begin(); it != nps.end(); ++it)
	{
		std::cout << it->key() << std::endl;
		std::cout << it->value() << std::endl;
	}
	std::cout << std::endl;
}

END_RLTL_IMPL