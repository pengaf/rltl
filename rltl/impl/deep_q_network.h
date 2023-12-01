#pragma once
#include "agent.h"
#include "epsilon_greedy.h"
#include "replay_memory.h"
#include "array.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct DeepQNetworkOptions
{
	DeepQNetworkOptions(float discountRate, size_t replayCapacity, size_t minReplaySize, size_t batchSize, size_t learnFreq, size_t targetUpdateFreq, bool doubleDQN) :
		m_discountRate(discountRate),
		m_replayCapacity(replayCapacity),
		m_minReplaySize(minReplaySize),
		m_batchSize(batchSize),
		m_learnFreq(learnFreq),
		m_targetUpdateFreq(targetUpdateFreq),
		m_doubleDQN(doubleDQN)
	{}
	RLTL_ARG(float, discountRate);
	RLTL_ARG(uint32_t, replayCapacity);
	RLTL_ARG(uint32_t, minReplaySize);
	RLTL_ARG(uint32_t, batchSize);
	RLTL_ARG(uint32_t, learnFreq);
	RLTL_ARG(uint32_t, targetUpdateFreq);
	RLTL_ARG(bool, doubleDQN);
};

template<typename ActionValueNet_t, typename Optimizer_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>, typename Reward_t = float>
class DeepQNetwork
{
public:
	typedef typename ActionValueNet_t::State_t State_t;
	typedef typename ActionValueNet_t::Value_t Value_t;
	typedef typename ActionValueNet_t::Action_t Action_t;
	typedef typename Reward_t Reward_t;
	typedef ReplayMemory<State_t, Action_t, Reward_t> ReplayMemory_t;
public:
	DeepQNetwork(ActionValueNet_t& valueNet, Optimizer_t& optimizer, Policy_t& policy, const DeepQNetworkOptions& options) :
		m_valueNet(valueNet),
		m_optimizer(optimizer),
		m_policy(policy),
		m_targetNet(*valueNet.get()),
		m_replayMemory(options.replayCapacity()),
		m_discountRate(options.discountRate()),
		m_minReplaySize(options.minReplaySize()),
		m_batchSize(options.batchSize()),
		m_learnFreq(options.learnFreq()),
		m_targetUpdateFreq(options.targetUpdateFreq()),
		m_doubleDQN(options.doubleDQN())
	{
		m_statesTensor = MakeTensor<State_t>(torch::kFloat32);
		m_actionsTensor = MakeTensor<Action_t>(torch::kInt64);
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
	Action_t nextStep(Reward_t reward, const State_t& nextState)
	{
		step(reward, nextState, true);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(Reward_t reward, const State_t& nextState, bool nonterminal)
	{
		step(reward, nextState, nonterminal);
	}
protected:
	void step(Reward_t reward, const State_t& nextState, bool nonterminal)
	{
		m_replayMemory.store(m_state, m_action, reward, nextState, nonterminal);
		if (m_replayMemory.size() < m_minReplaySize)
		{
			return;
		}
		++m_stepCount;// = (m_stepCount + 1) % m_learnFreq;
		if (m_stepCount % m_learnFreq == 0)
		{
			auto states = m_statesTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
			auto actions = m_actionsTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
			auto rewards = m_rewardsTensor.accessor<float, 2>();
			auto nextStates = m_nextStatesTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
			auto nonterminals = m_nonterminalsTensor.accessor<float, 2>();
			for (uint32_t i = 0; i < m_batchSize; ++i)
			{
				m_replayMemory.sample(states[i], actions[i], rewards[i], nextStates[i], nonterminals[i]);
			}
			Tensor valuesTensor = m_valueNet->forward(m_statesTensor).gather(1, m_actionsTensor);
			assert(valuesTensor.dim() == 2 && valuesTensor.size(0) == m_batchSize);

			Tensor maxNextValuesTensor;
			if (m_doubleDQN)
			{
				Tensor maxActionTensor = std::get<1>(m_valueNet->forward(m_nextStatesTensor).max(1, true));
				maxNextValuesTensor = m_targetNet->forward(m_nextStatesTensor).gather(1, maxActionTensor);
			}
			else
			{
				maxNextValuesTensor = std::get<0>(m_targetNet->forward(m_nextStatesTensor).max(1, true));
			}
			assert(maxNextValuesTensor.dim() == 2 && maxNextValuesTensor.size(0) == m_batchSize && maxNextValuesTensor.size(1) == 1);
			Tensor targetsTensor = m_rewardsTensor + maxNextValuesTensor * m_nonterminalsTensor * m_discountRate;
			assert(targetsTensor.dim() == 2 && targetsTensor.size(0) == m_batchSize && targetsTensor.size(1) == 1);
			Tensor lossTensor = torch::mean(torch::nn::functional::mse_loss(valuesTensor, targetsTensor));
			assert(lossTensor.dim() == 0 && 1 == lossTensor.numel());

			m_optimizer.zero_grad();
			lossTensor.backward();
			m_optimizer.step();

			if (m_valueNetUpdateCount % m_targetUpdateFreq == 0)
			{
				//static bool b = true;
				//if(b)
				//{
				//	b = false;
				//	NN_printParameters(m_targetNet);
				//	NN_printParameters(m_valueNet);
				//}
				NN_copyParameters(m_targetNet, m_valueNet);
				//std::cout << "LOSS: " << lossTensor << std::endl;
				//static bool b = true;
				//if (b)
				//{
				//	b = false;
				//	NN_printParameters(m_targetNet);
				//	m_targetNet->print();
				//}
			}
			++m_valueNetUpdateCount;
		}
	}

	template<typename Element_t, typename TensorScalar_t>
	Tensor MakeTensor(TensorScalar_t dtype)
	{
		auto shape = GetShape<Element_t>::shape();
		std::array<int64_t, GetDimension<Element_t>::dim() + 1> tensorShape;
		tensorShape[0] = m_batchSize;
		for (size_t i = 0; i < GetDimension<Element_t>::dim(); ++i)
		{
			tensorShape[i + 1] = shape[i];
		}
		Tensor tensor = torch::empty(tensorShape, torch::TensorOptions().dtype(dtype));
		return tensor;
	}

	//template<typename Array_t, typename TensorAccessor>
protected:
	ActionValueNet_t& m_valueNet;
	Optimizer_t& m_optimizer;
	Policy_t& m_policy;
	ActionValueNet_t m_targetNet;
	ReplayMemory_t m_replayMemory;
	float m_discountRate;
	uint32_t m_minReplaySize;
	uint32_t m_batchSize;
	uint32_t m_learnFreq;
	uint32_t m_stepCount{};
	uint32_t m_targetUpdateFreq;
	uint32_t m_valueNetUpdateCount{};
	bool m_doubleDQN;
	Tensor m_statesTensor;
	Tensor m_actionsTensor;
	Tensor m_rewardsTensor;
	Tensor m_nextStatesTensor;
	Tensor m_nonterminalsTensor;
protected:
	State_t m_state;
	Action_t m_action;
};

template<typename ActionValueNet_t, typename Optimizer_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>, typename Reward_t = float>
inline static DeepQNetwork<ActionValueNet_t, Optimizer_t, Policy_t, Reward_t> MakeDeepQNetwork(ActionValueNet_t& valueNet, Optimizer_t& optimizer, Policy_t& policy, const DeepQNetworkOptions& options)
{
	return DeepQNetwork(valueNet, optimizer, policy, options);
}

END_RLTL_IMPL