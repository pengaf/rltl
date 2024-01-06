#pragma once
#include "agent.h"
#include "trajectory_buffer.h"
#include "array.h"
#include "neural_network.h"
#include "multi_step_buffer.h"
#include "exploration.h"

BEGIN_RLTL_IMPL

struct DeepActionValueOptions
{
	DeepActionValueOptions(float discountRate, size_t batchSize) :
		m_discountRate(discountRate),
		m_batchSize(batchSize)
	{
		m_targetNetUpdateFreq = 0;// target network enabled if > 1
		m_multiStep = 0;// enabled if > 1
		m_multiStepCompound = false;
		m_experienceReplay = ExperienceReplay::no_experience_replay;
		m_replayMemorySize = 0;
		m_warmUpSize = 0;
		m_learnFreq = 0;
		m_prioritizedEpsilon = FLT_EPSILON;
		m_prioritizedAlpha = 1.0f;
		m_prioritizedBeta = 1.0f;
	}
public:
	DeepActionValueOptions& targetNetwork(uint32_t targetNetUpdateFreq)
	{
		m_targetNetUpdateFreq = targetNetUpdateFreq;
		return *this;
	}
	DeepActionValueOptions& multiStep(uint32_t step, bool compound)
	{
		m_multiStep = step;
		m_multiStepCompound = compound;
		return *this;
	}
	DeepActionValueOptions& experienceReplay(uint32_t replayMemorySize, uint32_t warmUpSize, size_t learnFreq)
	{
		m_experienceReplay = ExperienceReplay::experience_replay;
		m_replayMemorySize = replayMemorySize;
		m_warmUpSize = warmUpSize;
		m_learnFreq = learnFreq;
		return *this;
	}
	DeepActionValueOptions& prioritizedExperienceReplay(uint32_t replayMemorySize, uint32_t warmUpSize, size_t learnFreq, float prioritizedAlpha = 1.0f, float prioritizedBeta = 1.0f, float prioritizedEpsilon = FLT_EPSILON)
	{
		m_experienceReplay = ExperienceReplay::prioritized_experience_replay;
		m_replayMemorySize = replayMemorySize;
		m_warmUpSize = warmUpSize;
		m_learnFreq = learnFreq;
		m_prioritizedAlpha = prioritizedAlpha;
		m_prioritizedBeta = prioritizedBeta;
		m_prioritizedEpsilon = prioritizedEpsilon;
		return *this;
	}
public:
	RLTL_ARG(float, discountRate);
	RLTL_ARG(uint32_t, batchSize);
	RLTL_ARG(uint32_t, targetNetUpdateFreq);
	RLTL_ARG(uint32_t, multiStep);
	RLTL_ARG(bool, multiStepCompound);
	RLTL_ARG(ExperienceReplay, experienceReplay);
	RLTL_ARG(uint32_t, replayMemorySize);
	RLTL_ARG(uint32_t, warmUpSize);
	RLTL_ARG(uint32_t, learnFreq);
	RLTL_ARG(float, prioritizedEpsilon);
	RLTL_ARG(float, prioritizedAlpha);
	RLTL_ARG(float, prioritizedBeta);
};

struct DeepQLearningOptions : DeepActionValueOptions
{
	DeepQLearningOptions(float discountRate, size_t batchSize, bool doubleDQN) :
		DeepActionValueOptions(discountRate, batchSize),
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

struct DeepSarsaOptions : DeepActionValueOptions
{
	using DeepActionValueOptions::DeepActionValueOptions;
};

struct DeepSarsaExtData
{
public:
	DeepSarsaExtData(const DeepSarsaOptions& options)
	{}
public:
	Tensor m_nextActionTensor;
};

struct DeepExpectedSarsaOptions : DeepActionValueOptions
{
	using DeepActionValueOptions::DeepActionValueOptions;
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


template<TargetEvaluationMethod t_evaluationMethod, typename ActionValueNet_t, typename PolicyFunction_t = EpsilonGreedy<typename ActionValueNet_t::State_t, typename ActionValueNet_t::Action_t>>
class DeepQNetwork : public Agent<typename ActionValueNet_t::State_t, typename ActionValueNet_t::Action_t>, public DeepActionValueTraits<t_evaluationMethod>::ExtData
{
public:
	static_assert(TargetEvaluationMethod::q_learning == t_evaluationMethod || TargetEvaluationMethod::sarsa == t_evaluationMethod || TargetEvaluationMethod::expected_sarsa == t_evaluationMethod);
	//static_assert(SampleDepthCategory::td_0 == t_sampleDepthCategory || SampleDepthCategory::n_step_td == t_evaluationMethod || SampleDepthCategory::monte_carlo == t_evaluationMethod);
	typedef typename ActionValueNet_t::State_t State_t;
	typedef typename ActionValueNet_t::Action_t Action_t;

	typedef paf::SharedPtr<ActionValueNet_t> ActionValueNetPtr;
	typedef std::shared_ptr<Optimizer> OptimizerPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef typename DeepActionValueTraits<t_evaluationMethod>::Options Options;
	typedef paf::SharedPtr<DeepQNetwork> DeepQNetworkPtr;
public:
	DeepQNetwork(ActionValueNetPtr valueNet, OptimizerPtr optimizer, PolicyFunctionPtr policy, const Options& options) :
		DeepActionValueTraits<t_evaluationMethod>::ExtData(options),
		m_valueNet(valueNet),
		m_optimizer(optimizer),
		m_policy(policy),
		m_discountRate(options.discountRate()),
		m_batchSize(options.batchSize()),
		m_targetNetUpdateFreq(options.targetNetUpdateFreq()),
		m_multiStepCompound(options.multiStepCompound()),
		m_experienceReplay(options.experienceReplay()),
		m_replayMemorySize(options.replayMemorySize()),
		m_warmUpSize(options.warmUpSize()),
		m_learnFreq(options.learnFreq()),
		m_prioritizedEpsilon(options.prioritizedEpsilon()),
		m_prioritizedAlpha(options.prioritizedAlpha()),
		m_prioritizedBeta(options.prioritizedBeta())
	{
		m_targetNet = ActionValueNetPtr::Make(*valueNet->get());
		m_targetNet->get()->device(m_valueNet->get()->device());
		uint32_t multiStep = options.multiStep();
		if (multiStep > 1)
		{
			m_multiStepBuffer.initialize(multiStep);
		}
		if (m_learnFreq < 1)
		{
			m_learnFreq = 1;
		}
		if (useTargetNet())
		{
			NN_copyParameters(m_targetNet->get(), m_valueNet->get());
		}
		uint32_t bufferCapacity;
		if (ExperienceReplay::no_experience_replay == m_experienceReplay)
		{
			bufferCapacity = m_batchSize + multiStep * 2;
		}
		else
		{
			bufferCapacity = std::max(m_replayMemorySize, m_warmUpSize);
		}
		m_trajectoryBuffer.initialize(bufferCapacity, TargetEvaluationMethod::sarsa == t_evaluationMethod, ExperienceReplay::prioritized_experience_replay == m_experienceReplay);
		m_stateTensor = MakeTensor<State_t>(torch::kFloat32, m_batchSize);
		m_actionTensor = MakeTensor<Action_t>(torch::kInt64, m_batchSize);
		m_rewardTensor = MakeTensor<float>(torch::kFloat32, m_batchSize);
		m_nextStateTensor = MakeTensor<State_t>(torch::kFloat32, m_batchSize);
		m_nextDiscountTensor = MakeTensor<float>(torch::kFloat32, m_batchSize);
		if constexpr(TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			m_nextActionTensor = MakeTensor<Action_t>(torch::kInt64);
		}
		if (ExperienceReplay::prioritized_experience_replay == m_experienceReplay)
		{
			m_sampleIndices.resize(m_batchSize);
		}
	}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy->takeAction(firstState);
		m_multiStepBuffer.reset();
		return m_action;
	}
	Action_t nextStep(float reward, const State_t& nextState)	
	{
		if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			Action_t nextAction = m_policy->takeAction(nextState);
			if (m_multiStepBuffer)
			{
				m_multiStepBuffer.append(m_state, m_action, reward, m_discountRate);
				if (m_multiStepCompound)
				{
					uint32_t count = std::min(m_multiStepBuffer.stepCount(), m_multiStepBuffer.multiStep());
					for (size_t i = 0; i < count; ++i)
					{
						auto sar = m_multiStepBuffer.get(m_multiStepBuffer.stepCount() - count + i);
						m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, sar->accDiscountRate, nextAction);
					}
				}
				else
				{
					if (m_multiStepBuffer.stepCount() >= m_multiStepBuffer.multiStep())
					{
						auto sar = m_multiStepBuffer.getHead();
						m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, sar->accDiscountRate, nextAction);
					}
				}
			}
			else
			{
				m_trajectoryBuffer.append(m_state, m_action, reward, nextState, m_discountRate, nextAction);
			}
			learn(false);
			m_state = nextState;
			m_action = nextAction;
		}
		else
		{
			if (m_multiStepBuffer)
			{
				m_multiStepBuffer.append(m_state, m_action, reward, m_discountRate);
				if (m_multiStepCompound)
				{
					uint32_t count = std::min(m_multiStepBuffer.stepCount(), m_multiStepBuffer.multiStep());
					for (size_t i = 0; i < count; ++i)
					{
						auto sar = m_multiStepBuffer.get(m_multiStepBuffer.stepCount() - count + i);
						m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, sar->accDiscountRate);
					}
				}
				else
				{
					if (m_multiStepBuffer.stepCount() >= m_multiStepBuffer.multiStep())
					{
						auto sar = m_multiStepBuffer.getHead();
						m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, sar->accDiscountRate);
					}
				}
			}
			else
			{
				m_trajectoryBuffer.append(m_state, m_action, reward, nextState, m_discountRate);
			}
			learn(false);
			m_state = nextState;
			m_action = m_policy->takeAction(nextState);
		}
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			Action_t nextAction = m_policy->takeAction(nextState);
			if (m_multiStepBuffer)
			{
				m_multiStepBuffer.append(m_state, m_action, reward, m_discountRate);
				uint32_t count = std::min(m_multiStepBuffer.stepCount(), m_multiStepBuffer.multiStep());
				for (size_t i = 0; i < count; ++i)
				{
					auto sar = m_multiStepBuffer.get(m_multiStepBuffer.stepCount() - count + i);
					m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, terminated ? 0 : sar->accDiscountRate, nextAction);
				}
			}
			else
			{ 
				m_trajectoryBuffer.append(m_state, m_action, reward, nextState, terminated ? 0 : m_discountRate, nextAction);
			}
		}
		else
		{
			if (m_multiStepBuffer)
			{
				m_multiStepBuffer.append(m_state, m_action, reward, m_discountRate);
				uint32_t count = std::min(m_multiStepBuffer.stepCount(), m_multiStepBuffer.multiStep());
				for (size_t i = 0; i < count; ++i)
				{
					auto sar = m_multiStepBuffer.get(m_multiStepBuffer.stepCount() - count + i);
					m_trajectoryBuffer.append(sar->state, sar->action, sar->accReward, nextState, terminated ? 0 : sar->accDiscountRate);
				}
			}
			else
			{
				m_trajectoryBuffer.append(m_state, m_action, reward, nextState, terminated ? 0 : m_discountRate);
			}
		}
		learn(true);
	}
protected:
	bool useTargetNet() const
	{
		return m_targetNetUpdateFreq > 1;
	}
	void learn(bool lastStep)
	{
		++m_tryLearnCount;
		uint32_t batchSize = m_batchSize;
		if (ExperienceReplay::no_experience_replay == m_experienceReplay)
		{
			if (lastStep)
			{
				batchSize = m_trajectoryBuffer.size();
			}
			else if (m_trajectoryBuffer.size() < m_batchSize)
			{
				return;
			}
		}
		else
		{
			if (m_trajectoryBuffer.size() < m_warmUpSize)
			{
				return;
			}
			if (m_tryLearnCount % m_learnFreq != 0)
			{
				return;
			}
		}
		++m_learnCount;

		//Tensor stateTensor = m_stateTensor;
		//Tensor actionTensor = m_actionTensor;
		//Tensor rewardTensor = m_rewardTensor;
		//Tensor nextStateTensor = m_nextStateTensor;
		//Tensor nextDiscountTensor = m_nextDiscountTensor;
		//Tensor weightTensor = m_weightTensor;

		Tensor stateTensor = MakeTensor<State_t>(torch::kFloat32, batchSize);
		Tensor actionTensor = MakeTensor<Action_t>(torch::kInt64, batchSize);
		Tensor rewardTensor = MakeTensor<float>(torch::kFloat32, batchSize);
		Tensor nextStateTensor = MakeTensor<State_t>(torch::kFloat32, batchSize);
		Tensor nextDiscountTensor = MakeTensor<float>(torch::kFloat32, batchSize);
		Tensor weightTensor = MakeTensor<float>(torch::kFloat32, batchSize);


		if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			switch (m_experienceReplay)
			{
			case ExperienceReplay::no_experience_replay:
				m_trajectoryBuffer.pop(stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, nextActionTensor, batchSize);
				break;
			case ExperienceReplay::experience_replay:
				assert(batchSize == m_batchSize);
				m_trajectoryBuffer.sample(stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, nextActionTensor, batchSize);
				break;
			case ExperienceReplay::prioritized_experience_replay:
				assert(batchSize == m_batchSize);
				m_trajectoryBuffer.sample(m_sampleIndices, stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, nextActionTensor, weightTensor, batchSize, m_prioritizedBeta);
				break;
			default:
				return;
			}
		}
		else
		{
			switch (m_experienceReplay)
			{
			case ExperienceReplay::no_experience_replay:
				m_trajectoryBuffer.pop(stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, batchSize);
				break;
			case ExperienceReplay::experience_replay:
				assert(batchSize == m_batchSize);
				m_trajectoryBuffer.sample(stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, batchSize);
				break;
			case ExperienceReplay::prioritized_experience_replay:
				assert(batchSize == m_batchSize);
				m_trajectoryBuffer.sample(m_sampleIndices, stateTensor, actionTensor, rewardTensor, nextStateTensor, nextDiscountTensor, weightTensor, batchSize, m_prioritizedBeta);
				break;
			default:
				return;
			}
		}

		torch::Device device = m_valueNet->get()->device();
		stateTensor = stateTensor.to(device);
		actionTensor = actionTensor.to(device);
		rewardTensor = rewardTensor.to(device);
		nextStateTensor = nextStateTensor.to(device);
		nextDiscountTensor = nextDiscountTensor.to(device);
		weightTensor = weightTensor.to(device);

		Tensor valueTensor = m_valueNet->forward(stateTensor).gather(1, actionTensor);
		assert(valueTensor.dim() == 2 && valueTensor.size(0) == m_batchSize);

		Tensor nextValueTensor;
		if constexpr (TargetEvaluationMethod::q_learning == t_evaluationMethod)
		{
			if (m_doubleDQN)
			{
				Tensor maxActionTensor = std::get<1>(m_valueNet->forward(nextStateTensor).max(1, true));
				nextValueTensor = m_targetNet->forward(nextStateTensor).gather(1, maxActionTensor);
			}
			else
			{
				nextValueTensor = std::get<0>(m_targetNet->forward(nextStateTensor).max(1, true));
			}
		}
		else if constexpr (TargetEvaluationMethod::sarsa == t_evaluationMethod)
		{
			nextValueTensor = m_targetNet->forward(nextStateTensor).gather(1, nextActionTensor);//semi-gradient
		}
		else
		{
			nextValueTensor = torch::empty({ m_batchSize, 1 }, torch::TensorOptions().dtype(torch::kFloat32));
			auto nextValueAccessor = nextValueTensor.accessor<float, 2>();

			Tensor nextValuesTensor = m_targetNet->forward(nextStateTensor);
			assert(nextValuesTensor.dim() == 2 && nextValuesTensor.size(0) == m_batchSize);
			auto nextValuesAccessor = nextValuesTensor.accessor<float, 2>();

			size_t count = nextValuesTensor.size(1);
			std::vector<float> nextValues(count);
			for (size_t i = 0; i < m_batchSize; ++i)
			{
				for (size_t j = 0; j < count; ++j)
				{
					nextValues[j] = nextValuesAccessor[i][j];
				}
				nextValueAccessor[i][0] = m_policy.getExpectedValue(nextValues);
			}
		}
		assert(nextValueTensor.dim() == 2 && nextValueTensor.size(0) == m_batchSize && nextValueTensor.size(1) == 1);
		Tensor targetTensor = rewardTensor + nextValueTensor * nextDiscountTensor;

		assert(targetTensor.dim() == 2 && targetTensor.size(0) == m_batchSize && targetTensor.size(1) == 1);
		Tensor lossTensor;
		if (ExperienceReplay::prioritized_experience_replay == m_experienceReplay)
		{
			Tensor deltaTensor = targetTensor - valueTensor;
			assert(deltaTensor.dim() == 2 && deltaTensor.size(0) == m_batchSize && deltaTensor.size(1) == 1);
			m_trajectoryBuffer.updatePriorities(m_sampleIndices, deltaTensor, m_batchSize, m_prioritizedAlpha, m_prioritizedEpsilon);
			Tensor costTensor = torch::nn::functional::mse_loss(valueTensor, targetTensor, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)) * m_weightTensor;
			assert(costTensor.dim() == 2 && costTensor.size(0) == m_batchSize && costTensor.size(1) == 1);
			lossTensor = torch::mean(costTensor);
		}
		else
		{
			lossTensor = torch::mean(torch::nn::functional::mse_loss(valueTensor, targetTensor));
		}
		assert(lossTensor.dim() == 0 && 1 == lossTensor.numel());
		m_optimizer->zero_grad();
		lossTensor.backward();
		m_optimizer->step();
		if (m_learnCount % m_targetNetUpdateFreq == 0)
		{
			NN_copyParameters(m_targetNet->module(), m_valueNet->module());
		}
	}
protected:
	template<typename Element_t, typename TensorScalar_t>
	Tensor MakeTensor(TensorScalar_t dtype, uint32_t batchSize)
	{
		auto shape = Array_Shape<Element_t>::shape();
		std::array<int64_t, Array_Dimension<Element_t>::dim() + 1> tensorShape;
		tensorShape[0] = batchSize;
		for (size_t i = 0; i < Array_Dimension<Element_t>::dim(); ++i)
		{
			tensorShape[i + 1] = shape[i];
		}
		Tensor tensor = torch::empty(tensorShape, torch::TensorOptions().dtype(dtype));
		return tensor;
	}
protected:
	ActionValueNetPtr m_valueNet;
	OptimizerPtr m_optimizer;
	PolicyFunctionPtr m_policy;
	ActionValueNetPtr m_targetNet;

	float m_discountRate;
	uint32_t m_batchSize;
	uint32_t m_targetNetUpdateFreq;
	bool m_multiStepCompound;
	ExperienceReplay m_experienceReplay;
	uint32_t m_replayMemorySize;
	uint32_t m_warmUpSize;
	uint32_t m_learnFreq;
	float m_prioritizedEpsilon;
	float m_prioritizedAlpha;
	float m_prioritizedBeta;
	uint32_t m_tryLearnCount{};
	uint32_t m_learnCount{};

	State_t m_state;
	Action_t m_action;
	MultiStepBuffer<State_t, Action_t> m_multiStepBuffer;
	TrajectoryBuffer<State_t, Action_t> m_trajectoryBuffer;
	std::vector<uint32_t> m_sampleIndices;

	Tensor m_stateTensor;
	Tensor m_actionTensor;
	Tensor m_rewardTensor;
	Tensor m_nextStateTensor;
	Tensor m_nextDiscountTensor;
	Tensor m_weightTensor;

public:
	static DeepQNetworkPtr Make(ActionValueNetPtr valueNet, OptimizerPtr optimizer, PolicyFunctionPtr policy, const DeepQLearningOptions& options)
	{
		return DeepQNetworkPtr::Make(valueNet, optimizer, policy, options);
	}
};


template<typename ActionValueNet_t, typename PolicyFunction_t>
inline static paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::q_learning, ActionValueNet_t, PolicyFunction_t>> MakeDQN(
	paf::SharedPtr<ActionValueNet_t> valueNet,
	std::shared_ptr<Optimizer> optimizer,
	paf::SharedPtr<PolicyFunction_t> policy,
	const DeepQLearningOptions& options)
{
	return paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::q_learning, ActionValueNet_t, PolicyFunction_t>>::Make(valueNet, optimizer, policy, options);
}

template<typename ActionValueNet_t, typename PolicyFunction_t>
inline static paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::sarsa, ActionValueNet_t, PolicyFunction_t>> MakeDeepSarsa(
	paf::SharedPtr<ActionValueNet_t> valueNet,
	std::shared_ptr<Optimizer> optimizer,
	paf::SharedPtr<PolicyFunction_t> policy,
	const DeepQLearningOptions& options)
{
	return paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::sarsa, ActionValueNet_t, PolicyFunction_t>>::Make(valueNet, optimizer, policy, options);
}

template<typename ActionValueNet_t, typename PolicyFunction_t>
inline static paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ActionValueNet_t, PolicyFunction_t>> MakeDeepExpectedSarsa(
	paf::SharedPtr<ActionValueNet_t> valueNet,
	std::shared_ptr<Optimizer> optimizer,
	paf::SharedPtr<PolicyFunction_t> policy,
	const DeepQLearningOptions& options)
{
	return paf::SharedPtr<DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ActionValueNet_t, PolicyFunction_t>>::Make(valueNet, optimizer, policy, options);
}

//template<typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
//inline static DeepQNetwork<TargetEvaluationMethod::sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t> MakeDeepSarsa(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const DeepSarsaOptions& options)
//{
//	return DeepQNetwork<TargetEvaluationMethod::sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t>(replayMemory, valueNet, optimizer, policy, options);
//}
//
//template<typename ReplayMemory_t, typename ActionValueNet_t, typename Policy_t = EpsilonGreedy<ActionValueNet_t>>
//inline static DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t> MakeDeepExpectedSarsa(ReplayMemory_t& replayMemory, ActionValueNet_t& valueNet, Optimizer& optimizer, Policy_t& policy, const DeepExpectedSarsaOptions& options)
//{
//	return DeepQNetwork<TargetEvaluationMethod::expected_sarsa, ReplayMemory_t, ActionValueNet_t, Policy_t>(replayMemory, valueNet, optimizer, policy, options);
//}

END_RLTL_IMPL