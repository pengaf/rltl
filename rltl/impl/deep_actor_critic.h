#pragma once
#include "agent.h"
#include "epsilon_greedy.h"
#include "replay_memory.h"
#include "array.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct DeepActorCriticOptions
{
	DeepActorCriticOptions(float discountRate, size_t minReplaySize, size_t batchSize, size_t learnFreq) :
		m_discountRate(discountRate),
		m_minReplaySize(minReplaySize),
		m_batchSize(batchSize),
		m_learnFreq(learnFreq)
	{}
	RLTL_ARG(float, discountRate);
	RLTL_ARG(uint32_t, minReplaySize);
	RLTL_ARG(uint32_t, batchSize);
	RLTL_ARG(uint32_t, learnFreq);
};


template<typename ReplayMemory_t, typename PolicyNet_t, typename StateValueNet_t, typename Policy_t = PolicyNet_t>
class DeepActorCritic
{
public:
	typedef typename PolicyNet_t::State_t State_t;
	typedef typename PolicyNet_t::Action_t Action_t;
public:
	DeepActorCritic(ReplayMemory_t& replayMemory, PolicyNet_t& actorNet, Optimizer& actorOptimizer, StateValueNet_t& criticNet, Optimizer& criticOptimizer, Policy_t& policy, const DeepActorCriticOptions& options) :
		m_replayMemory(replayMemory),
		m_actorNet(actorNet),
		m_actorOptimizer(actorOptimizer),
		m_criticNet(criticNet),
		m_criticOptimizer(criticOptimizer),
		m_policy(policy),
		m_discountRate(options.discountRate()),
		m_minReplaySize(options.minReplaySize()),
		m_batchSize(options.batchSize()),
		m_learnFreq(options.learnFreq())
	{
		m_stateTensor = MakeTensor<State_t>(torch::kFloat32);
		m_actionTensor = MakeTensor<Action_t>(torch::kInt64);
		m_rewardTensor = MakeTensor<float>(torch::kFloat32);
		m_nextStateTensor = MakeTensor<State_t>(torch::kFloat32);
		m_nonterminalTensor = MakeTensor<bool>(torch::kFloat32);
		if (m_replayMemory.isPrioritized())
		{
			m_weightTensor = MakeTensor<float>(torch::kFloat32);
			m_sampleIndices.resize(m_batchSize);
		}
	}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(float reward, const State_t& nextState)
	{
		step(reward, nextState, true);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		step(reward, nextState, nonterminal);
	}
protected:
	void step(float reward, const State_t& nextState, bool nonterminal)
	{
		m_replayMemory.store(m_state, m_action, reward, nextState, nonterminal);
		if (m_replayMemory.size() < m_minReplaySize)
		{
			return;
		}
		++m_stepCount;// = (m_stepCount + 1) % m_learnFreq;
		if (m_stepCount % m_learnFreq == 0)
		{
			if (m_replayMemory.isPrioritized())
			{
				m_replayMemory.sample(m_sampleIndices, m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nonterminalTensor, m_weightTensor, m_batchSize);
			}
			else
			{
				m_replayMemory.sample(m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nonterminalTensor, m_batchSize);
			}

			Tensor valueTensor = m_criticNet->forward(m_stateTensor);
			assert(valueTensor.dim() == 2 && valueTensor.size(0) == m_batchSize && valueTensor.size(1) == 1);
			Tensor nextValueTensor = m_criticNet->forward(m_nextStateTensor);
			assert(nextValueTensor.dim() == 2 && nextValueTensor.size(0) == m_batchSize && nextValueTensor.size(1) == 1);
			Tensor targetTensor = m_rewardTensor + nextValueTensor * m_nonterminalTensor * m_discountRate;//n-step?
			assert(targetTensor.dim() == 2 && targetTensor.size(0) == m_batchSize && targetTensor.size(1) == 1);
			Tensor deltaTensor = valueTensor - targetTensor;//neg
			assert(deltaTensor.dim() == 2 && deltaTensor.size(0) == m_batchSize && deltaTensor.size(1) == 1);

			Tensor criticLossTensor;
			if (m_replayMemory.isPrioritized())
			{
				m_replayMemory.updatePriorities(m_sampleIndices, deltaTensor, m_batchSize);
				Tensor costTensor = torch::nn::functional::mse_loss(valueTensor, targetTensor, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)) * m_weightTensor;
				assert(costTensor.dim() == 2 && costTensor.size(0) == m_batchSize && costTensor.size(1) == 1);
				criticLossTensor = torch::mean(costTensor);
			}
			else
			{
				criticLossTensor = torch::mean(torch::nn::functional::mse_loss(valueTensor, targetTensor));
			}

			Tensor logProbTensor = torch::nn::functional::log_softmax(m_actorNet->forward(m_stateTensor), 1);
			Tensor actorLossTensor = logProbTensor.gather(1, m_actionTensor) * deltaTensor.detach();

			m_actorOptimizer.zero_grad();
			actorLossTensor.sum().backward();
			m_actorOptimizer.step();

			m_criticOptimizer.zero_grad();
			criticLossTensor.sum().backward();
			m_criticOptimizer.step();
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
	ReplayMemory_t& m_replayMemory;
	PolicyNet_t m_actorNet;
	Optimizer& m_actorOptimizer;
	StateValueNet_t& m_criticNet;
	Optimizer& m_criticOptimizer;
	Policy_t& m_policy;
	float m_discountRate;
	uint32_t m_minReplaySize;
	uint32_t m_batchSize;
	uint32_t m_learnFreq;
	uint32_t m_stepCount{};
	Tensor m_stateTensor;
	Tensor m_actionTensor;
	Tensor m_rewardTensor;
	Tensor m_nextStateTensor;
	Tensor m_nonterminalTensor;
	Tensor m_weightTensor;
	std::vector<uint32_t> m_sampleIndices;
protected:
	State_t m_state;
	Action_t m_action;
};

template<typename ReplayMemory_t, typename PolicyNet_t, typename StateValueNet_t, typename Policy_t = PolicyNet_t>
inline static DeepActorCritic<ReplayMemory_t, PolicyNet_t, StateValueNet_t, Policy_t> MakeDeepActorCritic(
	ReplayMemory_t& replayMemory, 
	PolicyNet_t& actorNet, 
	Optimizer& actorOptimizer, 
	StateValueNet_t& criticNet, 
	Optimizer& criticOptimizer, 
	Policy_t& policy, 
	const DeepActorCriticOptions& options)
{
	return DeepActorCritic(replayMemory, actorNet, actorOptimizer, criticNet, criticOptimizer, policy, options);
}

END_RLTL_IMPL