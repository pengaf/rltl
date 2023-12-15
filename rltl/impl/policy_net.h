#pragma once
#include "utility.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct MLPPolicyNetImpl : torch::nn::Module
{
public:
	MLPPolicyNetImpl(uint32_t stateDim, uint32_t actionDim, uint32_t hiddenDim, size_t numHiddens = 1)
	{
		std::vector<uint32_t> hiddenDims;
		hiddenDims.resize(numHiddens, hiddenDim);
		construct(stateDim, actionDim, hiddenDims);
	}
	MLPPolicyNetImpl(uint32_t stateDim, uint32_t actionDim, const std::vector<uint32_t>& hiddenDims)
	{
		construct(stateDim, actionDim, hiddenDims);
	}
	
	MLPPolicyNetImpl(const MLPPolicyNetImpl& other)
	{
		construct(other.m_stateDim, other.m_actionDim, other.m_hiddenDims);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		assert(m_linears.size() == (m_hiddenDims.size() + 1));
		size_t numHiddens = m_hiddenDims.size();
		for (size_t i = 0; i < numHiddens; ++i)
		{
			x = torch::relu(m_linears[i](x));
		}
		torch::Tensor al = m_linears[numHiddens](x);
		return al;
	}
	
	torch::Tensor logitAction(torch::Tensor x)
	{
		torch::NoGradGuard nograd;
		return forward(x);
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
	void construct(uint32_t stateDim, uint32_t actionDim, const std::vector<uint32_t>& hiddenDims)
	{
		m_stateDim = stateDim;
		m_actionDim = actionDim;
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
		m_linears.push_back(register_module("action_logit", torch::nn::Linear(inFeatures, actionDim)));
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
protected:
	std::vector<torch::nn::Linear> m_linears;
	uint32_t m_stateDim;
	uint32_t m_actionDim;
	std::vector<uint32_t> m_hiddenDims;
};


template<typename State_t, typename Action_t>
class MLPPolicyNet : public torch::nn::ModuleHolder<MLPPolicyNetImpl>
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	
public:
	using torch::nn::ModuleHolder<MLPPolicyNetImpl>::ModuleHolder;
public:
	Action_t takeAction(const State_t& state)
	{
		return NN_actionBySoftmax<decltype(*this), State_t, Action_t>(*this, state);
	}
	uint32_t actionCount() const
	{
		return impl_->actionDim();
	}
};

END_RLTL_IMPL
