#pragma once
#include "utility.h"
#include "../arg.h"
#include "neural_network.h"
#include <vector>

BEGIN_RLTL_IMPL

struct ReinforceOptions
{
	ReinforceOptions(float discountRate) :
		m_discountRate(discountRate)
	{}
	RLTL_ARG(float, discountRate);
};

template<typename PolicyNet_t, typename Policy_t = PolicyNet_t>
class DeepReinforce
{
public:
	typedef typename PolicyNet_t::State_t State_t;
	typedef typename PolicyNet_t::Action_t Action_t;
public:
	DeepReinforce(PolicyNet_t& policyNet, Optimizer& optimizer, Policy_t& policy, const ReinforceOptions& options) :
		m_policyNet(policyNet),
		m_optimizer(optimizer),
		m_policy(policy),
		m_discountRate(options.discountRate())
	{}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		return m_action;
	}

	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_states.emplace_back(m_state);
		m_actions.emplace_back(m_action);
		m_rewards.emplace_back(reward);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		m_states.emplace_back(m_state);
		m_actions.emplace_back(m_action);
		m_rewards.emplace_back(reward);
		
		
		uint32_t batchSize = m_states.size();
		Tensor stateTensor = MakeTensor<State_t>(torch::kFloat32, batchSize);
		Tensor actionTensor = MakeTensor<Action_t>(torch::kInt64, batchSize);
		Tensor returnTensor = MakeTensor<float>(torch::kFloat32, batchSize);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto returns = returnTensor.accessor<float, 2>();

		float g = 0;
		for (size_t i = 0; i < batchSize; ++i)
		{
			size_t index = batchSize - 1 - i;
			g = g * m_discountRate + m_rewards[index];
			returns[index][0] = -g;
			assign(states[index], m_states[index]);
			assign(actions[index], m_actions[index]);
		}

		Tensor logProbTensor = torch::nn::functional::log_softmax(m_policyNet->forward(stateTensor), 1);
		Tensor lossTensor = logProbTensor.gather(1, actionTensor) * returnTensor;

		m_optimizer.zero_grad();
		lossTensor.sum().backward();
		m_optimizer.step();
		
		

		//Tensor stateTensor = MakeTensor<State_t>(torch::kFloat32, 1);
		//Tensor actionTensor = MakeTensor<Action_t>(torch::kInt64, 1);
		//Tensor returnTensor = MakeTensor<float>(torch::kFloat32, 1);
		//auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		//auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();

		//float g = 0;
		//uint32_t batchSize = m_states.size();
		//m_optimizer.zero_grad();
		//for (size_t i = 0; i < batchSize; ++i)
		//{
		//	size_t index = batchSize - 1 - i;
		//	g = g * m_discountRate + m_rewards[index];
		//	
		//	assign(states[0], m_states[index]);
		//	assign(actions[0], m_actions[index]);

		//	Tensor logProbTensor = torch::nn::functional::log_softmax(m_policyNet->forward(stateTensor), 1);
		//	Tensor lossTensor = logProbTensor.gather(1, actionTensor) * -g;
		//	lossTensor.backward();
		//}
		//m_optimizer.step();


		m_states.clear();
		m_actions.clear();
		m_rewards.clear();
	}
protected:
	template<typename Element_t, typename TensorScalar_t>
	Tensor MakeTensor(TensorScalar_t dtype, uint32_t batchSize)
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
protected:
	PolicyNet_t& m_policyNet;
	Optimizer& m_optimizer;
	Policy_t& m_policy;
	float m_discountRate;
	std::vector<State_t> m_states;
	std::vector<Action_t> m_actions;
	std::vector<float> m_rewards;
protected:
	State_t m_state;
	Action_t m_action;
};

template<typename PolicyNet_t, typename Policy_t = PolicyNet_t>
inline static DeepReinforce<PolicyNet_t, Policy_t> MakeReinforce(PolicyNet_t& policyNet, Optimizer& optimizer, Policy_t& policy, const ReinforceOptions& options)
{
	return DeepReinforce(policyNet, optimizer, policy, options);
}

END_RLTL_IMPL
