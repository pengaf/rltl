#pragma once
#include "utility.h"
#include <torch/torch.h>

BEGIN_RLTL_IMPL

template<size_t hidden_layers = 1, bool dueling = true>
struct MLPQNet : torch::nn::Module
{
	enum { num_layers = hidden_layers + 1 + (dueling ? 1 : 0) };
	torch::nn::Linear m_linear[num_layers];

	MLPQNet(int64_t stateDim, int64_t actionDim, int64_t hiddenDim)
	{
		for (size_t i = 0; i < hidden_layers + 1; ++i)
		{
			int64_t inFeatures = (0 == i) ? stateDim : hiddenDim;
			int64_t outFeatures = (hidden_layers == i) ? actionDim : hiddenDim;
			m_linear[i] = torch::nn::Linear(inFeatures, outFeatures);
			char name[256];
			sprintf_s(name, "linear%d", i + 1);
			register_module("linear", m_linear[i]);
		}
	}
	MLPQNet(int64_t stateDim, int64_t actionDim, std::array<int64_t, hidden_layers> hiddenDims)
	{
		for (size_t i = 0; i < hidden_layers; ++i)
		{
			char name[256];
			sprintf_s(name, "linear%d", i + 1);
			int64_t inFeatures = (0 == i) ? stateDim : hiddenDims[i - 1];
			int64_t outFeatures = hiddenDims[i];
			m_linear[i] = register_module(name, torch::nn::Linear(inFeatures, outFeatures));
		}
		if constexpr (dueling)
		{
			m_linear[hidden_layers] = register_module("v", torch::nn::Linear(hiddenDims[hidden_layers - 1], 1));
			m_linear[hidden_layers + 1] = register_module("a", torch::nn::Linear(hiddenDims[hidden_layers - 1], actionDim));
		}
		else
		{
			m_linear[hidden_layers] = register_module("q", torch::nn::Linear(hiddenDims[hidden_layers - 1], actionDim));
		}
	}
	torch::Tensor forward(torch::Tensor x)
	{
		for (size_t i = 0; i < hidden_layers; ++i)
		{
			x = torch::relu(m_linear[i](x));
		}
		if constexpr (dueling)
		{
			torch::Tensor v = m_linear[hidden_layers](x);
			torch::Tensor a = m_linear[hidden_layers](x);
			torch::Tensor q = v + a - a.mean(1).view({ -1, 1 });
			return q;
		}
		else
		{
			torch::Tensor q = m_linear[hidden_layers](x);
			return q;
		}
	}
};


END_RLTL_IMPL
