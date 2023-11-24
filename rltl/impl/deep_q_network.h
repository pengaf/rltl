#pragma once
#include "agent.h"
#include "epsilon_greedy.h"
#include "replay_memory.h"
#include "../nn/tensor.h"

BEGIN_RLTL_IMPL

struct DeepQNetworkOptions
{
	DeepQNetworkOptions(float discountRate, size_t replayCapacity, size_t minReplaySize, size_t targetUpdateFreq, size_t batchSize) :
		m_discountRate(discountRate),
		m_replayCapacity(replayCapacity),
		m_minReplaySize(minReplaySize),
		m_targetUpdateFreq(targetUpdateFreq),
		m_batchSize(batchSize)
	{}
	RLTL_ARG(float, discountRate);
	RLTL_ARG(uint32_t, replayCapacity);
	RLTL_ARG(uint32_t, minReplaySize);
	RLTL_ARG(uint32_t, batchSize);
	RLTL_ARG(uint32_t, targetUpdateFreq);
};

template<typename ActionValueNet_t, typename Optimizer_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
class DeepQNetwork
{
public:
	typedef typename ActionValueFunction_t::State_t State_t;
	typedef typename ActionValueFunction_t::Value_t Reward_t;
	typedef typename ActionValueFunction_t::Action_t Action_t;
	typedef ReplayMemory<State_t, Action_t, Reward_t> ReplayMemory_t;
public:
	DeepQNetwork(const DeepQNetworkOptions& options, ActionValueNet_t& valueNet, Optimizer_t optimizer, Policy_t& policy) :
		m_valueNet(valueNet),
		m_optimizer(optimizer),
		m_targetNet(valueNet),
		m_policy(policy),
		m_replayMemory(options.replayCapacity()),
		m_learningRate(options.learningRate()),
		m_discountRate(options.discountRate()),
		m_minReplaySize(options.minReplaySize()),
		m_batchSize(options.batchSize()),
		m_targetUpdateFreq(options.targetUpdateFreq())
	{
		m_statesTensor = MakeTensor<State_t>(torch::kFloat32);
		m_actionsTensor = MakeTensor<Action_t>(torch::kInt32);
		m_rewardsTensor = MakeTensor<Reward_t>(torch::kFloat32);
		m_nextStatesTensor = MakeTensor<State_t>(torch::kFloat32);
		m_nonterminalsTensor = MakeTensor<bool>(torch::kFloat32);
	}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(const Reward_t& reward, const State_t& nextState)
	{
		step(reward, nextState, false);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(const Reward_t& reward, const State_t& nextState, bool terminated)
	{
		step(reward, nextState, terminated);
	}
protected:
	void step(const Reward_t& reward, const State_t& nextState, bool terminated)
	{
		torch::Tensor tensor = torch::empty({ 2,2 }, torch::TensorOptions().dtype(torch::kFloat32));

		m_replayMemory.store(m_state, m_action, reward, nextState, false);
		if (m_replayMemory.size() > m_minReplaySize)
		{
			State_t state;
			Action_t action;
			Reward_t reward;
			State_t nextState;
			bool terminated;

			auto statesAccessor = m_statesTensor.accessor<float, State_t::t_dims + 1>();
			auto actionsAccessor = m_actionsTensor.accessor<float, Action_t::t_dims + 1>();
			auto rewardsAccessor = m_rewardsTensor.accessor<float, 2>();
			auto nextStatesAccessor = m_nextStatesTensor.accessor<float, State_t::t_dims + 1>();
			auto m_nonterminalsTensor = m_nextStatesTensor.accessor<float, 2>();

			for (uint32_t i = 0; i < m_batchSize; ++i)
			{
				m_replayMemory.sample(state, action, reward, nextState, terminated);
				statesAccessor[i]
			}

			torch::tensor()
			nn::Tensor ten
			update();
		}
	}

	template<typename Array_t, typename TensorScalar_t>
	torch::Tensor MakeTensor(TensorScalar_t dtype)
	{
		auto shape = Array_t::shape();
		std::array<int64_t, State_t::t_dims + 1> tensorShape;
		tensorShape[0] = m_batchSize;
		for (size_t i = 0; i < State_t::t_dims; ++i)
		{
			tensorShape[i + 1] = shape[i];
		}
		torch::Tensor tensor = torch::empty(tensorShape, torch::TensorOptions().dtype(dtype));
		return tensor;
	}

	//template<typename Array_t, typename TensorAccessor>
protected:
	ActionValueNet_t& m_valueNet;
	Optimizer_t& m_optimizer;
	Policy_t& m_policy;
	ActionValueNet_t m_targetNet;
	ReplayMemory_t m_replayMemory;
	float m_learningRate;
	float m_discountRate;
	uint32_t m_minReplaySize;
	uint32_t m_targetUpdateFreq;
	uint32_t m_batchSize;
	torch::Tensor m_statesTensor;
	torch::Tensor m_actionsTensor;
	torch::Tensor m_rewardsTensor;
	torch::Tensor m_nextStatesTensor;
	torch::Tensor m_nonterminalsTensor;
protected:
	State_t m_state;
	Action_t m_action;
};

template<typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
inline static DeepQNetwork<ActionValueNet_t, Policy_t> MakeDeepQNetwork(ActionValueNet_t& valueNet, ActionValueNet_t& targetNet, Policy_t& policy, size_t replayMemorySize, float learningRate, float discountRate = 1.0f)
{
	return QLearning(valueNet, targetNet, policy, replayMemorySize, learningRate, discountRate);
}

END_RLTL_IMPL