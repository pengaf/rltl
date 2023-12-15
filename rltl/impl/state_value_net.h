#pragma once
#include "utility.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

//template<typename State_t = float>
//class StateValueNet
//{
//public:
//	virtual float getValue(const State_t& state) = 0;
//};

class MLPStateValueNetImpl : public torch::nn::Module
{
public:
	MLPStateValueNetImpl(uint32_t stateDim, uint32_t hiddenDim, size_t numHiddens = 1)
	{
		std::vector<uint32_t> hiddenDims;
		hiddenDims.resize(numHiddens, hiddenDim);
		construct(stateDim, hiddenDims);
	}
	MLPStateValueNetImpl(uint32_t stateDim, const std::vector<uint32_t>& hiddenDims)
	{
		construct(stateDim, hiddenDims);
	}
	MLPStateValueNetImpl(const MLPStateValueNetImpl& other)
	{
		construct(other.m_stateDim, other.m_hiddenDims);
	}
	torch::Tensor forward(torch::Tensor x)
	{
		assert(m_linears.size() == (m_hiddenDims.size() + 1));
		size_t numHiddens = m_hiddenDims.size();
		for (size_t i = 0; i < numHiddens; ++i)
		{
			x = torch::relu(m_linears[i](x));
		}
		torch::Tensor v = m_linears[numHiddens](x);
		return v;
	}
	void print()
	{
		for (auto& layer : m_linears)
		{
			std::cout << layer << std::endl;
			for (auto& params : layer->parameters())
			{
				std::cout << params << std::endl;
			}
		}
	}
protected:
	void construct(uint32_t stateDim, const std::vector<uint32_t>& hiddenDims)
	{
		m_stateDim = stateDim;
		m_hiddenDims = hiddenDims;

		size_t numHiddens = hiddenDims.size();
		size_t numLayers = numHiddens + 1;
		m_linears.reserve(numLayers);

		uint32_t inFeatures = stateDim;
		for (size_t i = 0; i < numHiddens; ++i)
		{
			char name[256];
			sprintf_s(name, "linear_%d", i + 1);
			uint32_t outFeatures = hiddenDims[i];
			m_linears.push_back(register_module(name, torch::nn::Linear(inFeatures, outFeatures)));
			inFeatures = outFeatures;
		}
		inFeatures = numHiddens > 0 ? hiddenDims[numHiddens - 1] : stateDim;
		m_linears.push_back(register_module("state_value", torch::nn::Linear(inFeatures, 1)));
	}
public:
	uint32_t stateDim() const
	{
		return m_stateDim;
	}
protected:
	std::vector<torch::nn::Linear> m_linears;
	uint32_t m_stateDim;
	std::vector<uint32_t> m_hiddenDims;
};

template<typename State_t = float>
class MLPStateValueNet : public torch::nn::ModuleHolder<MLPStateValueNetImpl>
{
public:
	typedef State_t State_t;
	
public:
	using torch::nn::ModuleHolder<MLPStateValueNetImpl>::ModuleHolder;
public:
	float getValue(const State_t& state)
	{
		return NN_getStateValue<decltype(*this), State_t, float>(*this, state);
	}
};

END_RLTL_IMPL
