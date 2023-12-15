#pragma once
#include "utility.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct MLPPolicyStateValueValueNetImpl : torch::nn::Module
{
public:
	MLPPolicyStateValueValueNetImpl(
		uint32_t stateDim, 
		uint32_t actionDim, 
		uint32_t hiddenDim, 
		uint32_t numSharedHiddens = 1, 
		uint32_t numActionHiddens = 0, 
		uint32_t numStateValueHiddens = 0)
	{
		std::vector<uint32_t> sharedHiddenDims;
		sharedHiddenDims.resize(numSharedHiddens, hiddenDim);
		std::vector<uint32_t> actionHiddenDims;
		actionHiddenDims.resize(numActionHiddens, hiddenDim);
		std::vector<uint32_t> stateValueHiddenDims;
		stateValueHiddenDims.resize(numStateValueHiddens, hiddenDim);
		construct(stateDim, actionDim, sharedHiddenDims, actionHiddenDims, stateValueHiddenDims);
	}

	MLPPolicyStateValueValueNetImpl(
		uint32_t stateDim, 
		uint32_t actionDim, 
		const std::vector<uint32_t>& sharedHiddenDims, 
		const std::vector<uint32_t>& actionHiddenDims, 
		const std::vector<uint32_t>& stateValueHiddenDims)
	{
		construct(stateDim, actionDim, sharedHiddenDims, actionHiddenDims, stateValueHiddenDims);
	}

	MLPPolicyStateValueValueNetImpl(const MLPPolicyStateValueValueNetImpl& other)
	{
		construct(other.m_stateDim, other.m_actionDim, other.m_sharedHiddenDims, other.m_actionHiddenDims, other.m_stateValueHiddenDims);
	}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
	{
		size_t numSharedHiddens = m_sharedHiddenDims.size();
		size_t numActionHiddens = m_actionHiddenDims.size();
		size_t numStateValueHiddens = m_stateValueHiddenDims.size();
		for (size_t i = 0; i < numSharedHiddens; ++i)
		{
			x = torch::relu(m_linears[i](x));
		}
		torch::Tensor ah = x;
		for (size_t i = 0; i < numActionHiddens; ++i)
		{
			ah = torch::relu(m_linears[numSharedHiddens + i](ah));
		}
		torch::Tensor vh = x;
		for (size_t i = 0; i < numStateValueHiddens; ++i)
		{
			vh = torch::relu(m_linears[numSharedHiddens + numActionHiddens + i](vh));
		}
		torch::Tensor al = m_linears[numSharedHiddens + numActionHiddens + numStateValueHiddens](ah);
		torch::Tensor sv = m_linears[numSharedHiddens + numActionHiddens + numStateValueHiddens + 1](vh);
		return std::make_pair(al, sv);
	}

	torch::Tensor logitAction(torch::Tensor x)
	{
		torch::NoGradGuard nograd;
		return forward(x).first;
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
	void construct(uint32_t stateDim, uint32_t actionDim, const std::vector<uint32_t>& sharedHiddenDims, const std::vector<uint32_t>& actionHiddenDims, const std::vector<uint32_t>& stateValueHiddenDims)
	{
		m_stateDim = stateDim;
		m_actionDim = actionDim;
		m_sharedHiddenDims = sharedHiddenDims;
		m_actionHiddenDims = actionHiddenDims;
		m_stateValueHiddenDims = stateValueHiddenDims;

		size_t numSharedHiddens = sharedHiddenDims.size();
		size_t numActionHiddens = actionHiddenDims.size();
		size_t numStateValueHiddens = stateValueHiddenDims.size();

		size_t numLayers = numSharedHiddens + numActionHiddens + numStateValueHiddens + 2;
		m_linears.reserve(numLayers);

		uint32_t sharedInFeatures = stateDim;
		for (size_t i = 0; i < numSharedHiddens; ++i)
		{
			char name[256];
			sprintf_s(name, "shared_linear_%d", i + 1);
			uint32_t outFeatures = sharedHiddenDims[i];
			m_linears.push_back(register_module(name, torch::nn::Linear(sharedInFeatures, outFeatures)));
			sharedInFeatures = outFeatures;
		}
		uint32_t actionInFeatures = sharedInFeatures;
		for (size_t i = 0; i < numActionHiddens; ++i)
		{
			char name[256];
			sprintf_s(name, "action_linear_%d", i + 1);
			uint32_t outFeatures = actionHiddenDims[i];
			m_linears.push_back(register_module(name, torch::nn::Linear(actionInFeatures, outFeatures)));
			actionInFeatures = outFeatures;
		}
		uint32_t stateValueInFeatures = sharedInFeatures;
		for (size_t i = 0; i < numStateValueHiddens; ++i)
		{
			char name[256];
			sprintf_s(name, "state_value_linear_%d", i + 1);
			uint32_t outFeatures = stateValueHiddenDims[i];
			m_linears.push_back(register_module(name, torch::nn::Linear(stateValueInFeatures, outFeatures)));
			stateValueInFeatures = outFeatures;
		}
		m_linears.push_back(register_module("action_logit", torch::nn::Linear(actionInFeatures, actionDim)));
		m_linears.push_back(register_module("state_value", torch::nn::Linear(stateValueInFeatures, 1)));
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
	std::vector<uint32_t> m_sharedHiddenDims;
	std::vector<uint32_t> m_actionHiddenDims;
	std::vector<uint32_t> m_stateValueHiddenDims;
};


template<typename State_t, typename Action_t>
class MLPPolicyStateValueValueNet : public torch::nn::ModuleHolder<MLPPolicyStateValueValueNetImpl>
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
public:
	using torch::nn::ModuleHolder<MLPPolicyStateValueValueNetImpl>::ModuleHolder;
public:
	Action_t takeAction(const StateValue_t& stateValue)
	{
		return NN_actionBySoftmax<decltype(*this), State_t, Action_t>(*this, stateValue);
	}
};

END_RLTL_IMPL
