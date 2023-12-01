#pragma once
#include "utility.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct MLPQNetImpl : torch::nn::Module 
{
public:
	MLPQNetImpl(uint32_t stateDim, uint32_t actionDim, uint32_t hiddenDim, size_t numHiddens = 1, bool dueling = true)
	{
		std::vector<uint32_t> hiddenDims;
		hiddenDims.resize(numHiddens, hiddenDim);
		construct(stateDim, actionDim, hiddenDims, dueling);
	}
	MLPQNetImpl(uint32_t stateDim, uint32_t actionDim, const std::vector<uint32_t>& hiddenDims, bool dueling = true)
	{
		construct(stateDim, actionDim, hiddenDims, dueling);
	}
	MLPQNetImpl(const MLPQNetImpl& other)
	{
		construct(other.m_stateDim, other.m_actionDim, other.m_hiddenDims, other.m_dueling);
	}
	torch::Tensor forward(torch::Tensor x)
	{
		assert(m_linears.size() == (m_hiddenDims.size() + 1 + (m_dueling ? 1 : 0)));
		size_t numHiddens = m_hiddenDims.size();
		for (size_t i = 0; i < numHiddens; ++i)
		{
			x = torch::relu(m_linears[i](x));
		}
		if (m_dueling)
		{
			torch::Tensor v = m_linears[numHiddens](x);
			torch::Tensor a = m_linears[numHiddens + 1](x);
			torch::Tensor q = v + a - a.mean(1).view({ -1, 1 });
			return q;
		}
		else
		{
			torch::Tensor q = m_linears[numHiddens](x);
			return q;
		}
	}
	void print()
	{
		for (auto& l : m_linears)
		{
			std::cout << l << std::endl;
			for (auto& p : l->parameters())
			{
				std::cout << p << std::endl;
			}
		}
	}
protected:
	void construct(uint32_t stateDim, uint32_t actionDim, const std::vector<uint32_t>& hiddenDims, bool dueling)
	{
		m_stateDim = stateDim;
		m_actionDim = actionDim;
		m_hiddenDims = hiddenDims;
		m_dueling = dueling;

		size_t numHiddens = hiddenDims.size();
		size_t numLayers = numHiddens + 1 + (dueling ? 1 : 0);
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
		if (dueling)
		{
			m_linears.push_back(register_module("state_value", torch::nn::Linear(inFeatures, 1)));
			m_linears.push_back(register_module("advantage", torch::nn::Linear(inFeatures, actionDim)));
		}
		else
		{
			m_linears.push_back(register_module("action_value", torch::nn::Linear(inFeatures, actionDim)));
		}
	}
public:
	uint32_t stateDim() const
	{
		return m_stateDim;
	}
	uint32_t actionDim() const
	{
		return m_actionDim;
	}
	bool dueling() const
	{
		return m_dueling;
	}
protected:
	std::vector<torch::nn::Linear> m_linears;
	uint32_t m_stateDim;
	uint32_t m_actionDim;
	std::vector<uint32_t> m_hiddenDims;
	bool m_dueling;
};

//TORCH_MODULE(MLPQNet);

template<typename State_t, typename Action_t, typename Value_t = float>
class MLPQNet : public torch::nn::ModuleHolder<MLPQNetImpl> 
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Value_t Value_t;
public:
	using torch::nn::ModuleHolder<MLPQNetImpl>::ModuleHolder;
public:
	Action_t firstMaxAction(const State_t& state)
	{
		return NN_firstMaxAction<decltype(*this), State_t, Action_t>(*this, state);
	}
	Action_t randomMaxAction(const State_t& state)
	{
		assert(state < m_stateCount);
		return firstMaxAction(state);
	}
	uint32_t actionCount() const
	{
		return impl_->actionDim();
		//return m_actionCount;
	}
	Value_t firstMaxValue(const State_t& state) const	
	{
		return Value_t();
	}
};


END_RLTL_IMPL