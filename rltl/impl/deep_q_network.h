#pragma once
#include "agent.h"
#include "epsilon_greedy.h"
#include "replay_memory.h"
#include "array.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

struct DeepActionValueAgentOptions
{
	DeepActionValueAgentOptions(float discountRate, size_t minReplaySize, size_t batchSize, size_t learnFreq, size_t targetUpdateFreq) :
		m_discountRate(discountRate),
		m_minReplaySize(minReplaySize),
		m_batchSize(batchSize),
		m_learnFreq(learnFreq),
		m_targetUpdateFreq(targetUpdateFreq)
	{
		m_multiStep = 0;
	}
	RLTL_ARG(float, discountRate);
	RLTL_ARG(uint32_t, multiStep);
	RLTL_ARG(uint32_t, minReplaySize);
	RLTL_ARG(uint32_t, batchSize);
	RLTL_ARG(uint32_t, learnFreq);
	RLTL_ARG(uint32_t, targetUpdateFreq);
};

struct DeepQLearningOptions : DeepActionValueAgentOptions
{
	DeepQLearningOptions(float discountRate, size_t minReplaySize, size_t batchSize, size_t learnFreq, size_t targetUpdateFreq, bool doubleDQN) :
		DeepActionValueAgentOptions(discountRate, minReplaySize, batchSize, learnFreq, targetUpdateFreq),
		m_doubleDQN(doubleDQN)
	{}
	RLTL_ARG(bool, doubleDQN);
};

struct DeepQLearningExtData
{
public:
	DeepQLearningExtData(const DeepQLearningOptions& options) :
		m_doubleDQN(options.doubleDQN())
	{}
public:
	bool m_doubleDQN;
};

struct DeepSarsaOptions : DeepActionValueAgentOptions
{
	using DeepActionValueAgentOptions::DeepActionValueAgentOptions;
};

struct DeepSarsaExtData
{
public:
	DeepSarsaExtData(const DeepSarsaOptions& options)
	{}
public:
	Tensor m_nextActionTensor;
};

struct DeepExpectedSarsaOptions : DeepActionValueAgentOptions
{
	using DeepActionValueAgentOptions::DeepActionValueAgentOptions;
};

struct DeepExpectedSarsaExtData
{
public:
	DeepExpectedSarsaExtData(const DeepExpectedSarsaOptions& options)
	{}
};

template<TargetEvaluationMethod t_evaluationMethod>
struct DeepActionValueTraits
{};

template<>
struct DeepActionValueTraits<TargetEvaluationMethod::q_learning>
{
	typedef DeepQLearningOptions Options;
	typedef DeepQLearningExtData ExtData;
};

template<>
struct DeepActionValueTraits<TargetEvaluationMethod::sarsa>
{
	typedef DeepSarsaOptions Options;
	typedef DeepSarsaExtData ExtData;
};

template<>
struct DeepActionValueTraits<TargetEvaluationMethod::expected_sarsa>
{
	typedef DeepExpectedSarsaOptions Options;
	typedef DeepExpectedSarsaExtData ExtData;
};


template<TargetEvaluationMethod t_evaluationMethod, typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
class DeepQNetwork : public DeepActionValueTraits<t_evaluationMethod>::ExtData
{
public:
	static_assert(TargetEvaluationMethod::q_learning == t_evaluationMethod || TargetEvaluationMethod::sarsa == t_evaluationMethod || TargetEvaluationMethod::expected_sarsa == t_evaluationMethod);
	//static_assert(SampleDepthCategory::td_0 == t_sampleDepthCategory || SampleDepthCategory::n_step_td == t_evaluationMethod || SampleDepthCategory::monte_carlo == t_evaluationMethod);
	typedef typename ActionValueNet_t::State_t State_t;
	typedef typename ActionValueNet_t::Action_t Action_t;
	typedef typename DeepActionValueTraits<t_evaluationMethod>::Options Options;
public:
	DeepQNetwork(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const Options& options) :
		DeepActionValueTraits<t_evaluationMethod>::ExtData(options),
		m_valueNet(valueNet),
		m_optimizer(optimizer),
		m_policy(policy),
		m_targetNet(*valueNet.get()),
		m_replayMemory(replayMemory),
		m_discountRate(options.discountRate()),
		m_multiStep(options.multiStep()),
		m_minReplaySize(options.minReplaySize()),
		m_batchSize(options.batchSize()),
		m_learnFreq(options.learnFreq()),
		m_targetUpdateFreq(options.targetUpdateFreq())
	{
		m_stateTensor = MakeTensor<State_t>(torch::kFloat32);
		m_actionTensor = MakeTensor<Action_t>(torch::kInt64);
		m_rewardTensor = MakeTensor<float>(torch::kFloat32);
		m_nextStateTensor = MakeTensor<State_t>(torch::kFloat32);
		if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			replayMemory.prepareNextActions();
			m_nextActionTensor = MakeTensor<Action_t>(torch::kInt64);
		}
		m_nonterminalTensor = MakeTensor<bool>(torch::kFloat32);
		if (m_replayMemory.isPrioritized())
		{
			m_weightTensor = MakeTensor<float>(torch::kFloat32);
			m_sampleIndices.resize(m_batchSize);
		}
		if (m_multiStep > 1)
		{
			m_multiStepStates = new State_t[m_multiStep];
			m_multiStepActions = new Action_t[m_multiStep];
			m_multiStepRewards = new float[m_multiStep];
			m_multiStepDiscountRate = std::pow(m_discountRate, m_multiStep);
		}
		else
		{
			m_multiStepDiscountRate = m_discountRate;
		}
	}
	~DeepQNetwork()
	{
		delete[] m_multiStepStates;
		delete[] m_multiStepActions;
		delete[] m_multiStepRewards;
	}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		m_currentStep = 0;//multistep
		return m_action;
	}
	Action_t nextStep(float reward, const State_t& nextState)	
	{
		float accReward = 0;
		if (m_multiStep > 1)
		{
			uint32_t index = m_currentStep % m_multiStep;
			m_multiStepStates[index] = m_state;
			m_multiStepActions[index] = m_action;
			m_multiStepRewards[index] = reward;
			if (m_currentStep + 1 >= m_multiStep)
			{
				accReward = reward;
				for (uint32_t i = 1; i < m_multiStep; ++i)
				{
					accReward = accReward * m_discountRate + m_multiStepRewards[(m_currentStep - i) % m_multiStep];
				}
			}
		}
		if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			Action_t nextAction = m_policy.takeAction(nextState);
			if (m_multiStep > 1)
			{
				if (m_currentStep + 1 >= m_multiStep)
				{
					uint32_t index = (m_currentStep + 1) % m_multiStep;
					m_replayMemory.store(m_multiStepStates[index], m_multiStepActions[index], accReward, nextState, nextAction, true);
				}
			}
			else
			{
				m_replayMemory.store(m_state, m_action, reward, nextState, nextAction, true);
			}
			learn();
			m_state = nextState;
			m_action = nextAction;
		}
		else
		{
			if (m_multiStep > 1)
			{
				if (m_currentStep + 1 >= m_multiStep)
				{
					uint32_t index = (m_currentStep + 1) % m_multiStep;
					m_replayMemory.store(m_multiStepStates[index], m_multiStepActions[index], accReward, nextState, true);
				}
			}
			else
			{
				m_replayMemory.store(m_state, m_action, reward, nextState, true);
			}
			learn();
			m_state = nextState;
			m_action = m_policy.takeAction(nextState);
		}
		++m_currentStep;
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		if (m_multiStep > 1)
		{
			uint32_t index = m_currentStep % m_multiStep;
			m_multiStepStates[index] = m_state;
			m_multiStepActions[index] = m_action;
			m_multiStepRewards[index] = reward;
			size_t count = m_currentStep < m_multiStep ? m_currentStep : m_multiStep;
			if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
			{
				Action_t nextAction = m_policy.takeAction(nextState);
				float accReward = 0;
				for (size_t i = 0; i < count; ++i)
				{
					uint32_t index = (m_currentStep - i) % m_multiStep;
					accReward = accReward * m_discountRate + m_multiStepRewards[index];
					//if (!nonterminal || i + 1 == m_multiStep)
					{
						m_replayMemory.store(m_multiStepStates[index], m_multiStepActions[index], accReward, nextState, nextAction, nonterminal);
					}
				}
			}
			else
			{
				float accReward = 0;
				for (size_t i = 0; i < count; ++i)
				{
					uint32_t index = (m_currentStep - i) % m_multiStep;
					accReward = accReward * m_discountRate + m_multiStepRewards[index];
					//if (!nonterminal || i + 1 == m_multiStep)
					{
						m_replayMemory.store(m_multiStepStates[index], m_multiStepActions[index], accReward, nextState, nonterminal);
					}
				}
			}
		}
		else
		{
			if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
			{
				Action_t nextAction = m_policy.takeAction(nextState);
				m_replayMemory.store(m_state, m_action, reward, nextState, nextAction, nonterminal);
			}
			else
			{
				m_replayMemory.store(m_state, m_action, reward, nextState, nonterminal);
			}
		}
		learn();
	}
protected:
	void learn()
	{
		if (m_replayMemory.size() < m_minReplaySize)
		{
			return;
		}
		++m_stepCount;// = (m_stepCount + 1) % m_learnFreq;
		if (m_stepCount % m_learnFreq == 0)
		{
			if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
			{
				if (m_replayMemory.isPrioritized())
				{
					m_replayMemory.sample(m_sampleIndices, m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nextActionTensor, m_nonterminalTensor, m_weightTensor, m_batchSize);
				}
				else
				{
					m_replayMemory.sample(m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nextActionTensor, m_nonterminalTensor, m_batchSize);
				}
			}
			else
			{
				if (m_replayMemory.isPrioritized())
				{
					m_replayMemory.sample(m_sampleIndices, m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nonterminalTensor, m_weightTensor, m_batchSize);
				}
				else
				{
					m_replayMemory.sample(m_stateTensor, m_actionTensor, m_rewardTensor, m_nextStateTensor, m_nonterminalTensor, m_batchSize);
				}
			}

			Tensor valueTensor = m_valueNet->forward(m_stateTensor).gather(1, m_actionTensor);
			assert(valueTensor.dim() == 2 && valueTensor.size(0) == m_batchSize);

			Tensor nextValueTensor;
			if constexpr (TargetEvaluationMethod::q_learning == t_evaluationMethod)
			{
				if (m_doubleDQN)
				{
					Tensor maxActionTensor = std::get<1>(m_valueNet->forward(m_nextStateTensor).max(1, true));
					nextValueTensor = m_targetNet->forward(m_nextStateTensor).gather(1, maxActionTensor);
				}
				else
				{
					nextValueTensor = std::get<0>(m_targetNet->forward(m_nextStateTensor).max(1, true));
				}
			}
			else if constexpr(TargetEvaluationMethod::sarsa == t_evaluationMethod)
			{
				nextValueTensor = m_targetNet->forward(m_nextStateTensor).gather(1, m_nextActionTensor);//semi-gradient
			}
			else
			{
				nextValueTensor = torch::empty({m_batchSize, 1}, torch::TensorOptions().dtype(torch::kFloat32));
				auto nextValueAccessor = nextValueTensor.accessor<float, 2>();

				Tensor nextValuesTensor = m_targetNet->forward(m_nextStateTensor);
				assert(nextValuesTensor.dim() == 2 && nextValuesTensor.size(0) == m_batchSize);
				auto nextValuesAccessor = nextValuesTensor.accessor<float, 2>();

				size_t count = nextValuesTensor.size(1);
				std::vector<float> nextValues(count);
				for(size_t i = 0; i < m_batchSize; ++i)
				{
					for (size_t j = 0; j < count; ++j)
					{
						nextValues[j] = nextValuesAccessor[i][j];
					}
					nextValueAccessor[i][0] = m_policy.getExpectedValue(nextValues);
				}
			}
			assert(nextValueTensor.dim() == 2 && nextValueTensor.size(0) == m_batchSize && nextValueTensor.size(1) == 1);
			Tensor targetTensor = m_rewardTensor + nextValueTensor * m_nonterminalTensor * m_multiStepDiscountRate;

			assert(targetTensor.dim() == 2 && targetTensor.size(0) == m_batchSize && targetTensor.size(1) == 1);
			Tensor lossTensor;
			if (m_replayMemory.isPrioritized())
			{
				Tensor deltaTensor = targetTensor - valueTensor;
				assert(deltaTensor.dim() == 2 && deltaTensor.size(0) == m_batchSize && deltaTensor.size(1) == 1);
				m_replayMemory.updatePriorities(m_sampleIndices, deltaTensor, m_batchSize);
				Tensor costTensor = torch::nn::functional::mse_loss(valueTensor, targetTensor, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)) * m_weightTensor;
				assert(costTensor.dim() == 2 && costTensor.size(0) == m_batchSize && costTensor.size(1) == 1);
				lossTensor = torch::mean(costTensor);
			}
			else
			{
				lossTensor = torch::mean(torch::nn::functional::mse_loss(valueTensor, targetTensor));
			}
			assert(lossTensor.dim() == 0 && 1 == lossTensor.numel());
			m_optimizer.zero_grad();
			lossTensor.backward();
			m_optimizer.step();
			if (m_valueNetUpdateCount % m_targetUpdateFreq == 0)
			{
				NN_copyParameters(m_targetNet, m_valueNet);
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
	Optimizer& m_optimizer;
	Policy_t& m_policy;
	ActionValueNet_t m_targetNet;
	ReplayMemory_t& m_replayMemory;
	float m_discountRate;
	uint32_t m_multiStep;
	uint32_t m_minReplaySize;
	uint32_t m_batchSize;
	uint32_t m_learnFreq;
	uint32_t m_stepCount{};
	uint32_t m_targetUpdateFreq;
	uint32_t m_valueNetUpdateCount{};
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
protected:
	//for multi step
	State_t* m_multiStepStates{};
	Action_t* m_multiStepActions{};
	float* m_multiStepRewards{};
	uint32_t m_currentStep{};
	float m_multiStepDiscountRate{};
};

template<typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
inline static DeepQNetwork<TargetEvaluationMethod::q_learning, ReplayMemory_t, ActionValueNet_t, Policy_t> MakeDQN(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const DeepQLearningOptions& options)
{
	return DeepQNetwork<TargetEvaluationMethod::q_learning, ReplayMemory_t, ActionValueNet_t, Policy_t>(replayMemory, valueNet, optimizer, policy, options);
}

template<typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
inline static DeepQNetwork<TargetEvaluationMethod::sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t> MakeDeepSarsa(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const DeepSarsaOptions& options)
{
	return DeepQNetwork<TargetEvaluationMethod::sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t>(replayMemory, valueNet, optimizer, policy, options);
}

template<typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
inline static DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t> MakeDeepExpectedSarsa(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const DeepExpectedSarsaOptions& options)
{
	return DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t>(replayMemory, valueNet, optimizer, policy, options);
}

END_RLTL_IMPL