#pragma once
#include "agent.h"
#include "trajectory_buffer.h"
#include "array.h"
#include "neural_network.h"
#include "multi_step_buffer.h"
#include <assert.h>

BEGIN_RLTL_IMPL

struct DeepActorCriticOptions
{
	DeepActorCriticOptions(float discountRate, size_t batchSize) :
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
	DeepActorCriticOptions& targetNetwork(uint32_t targetNetUpdateFreq)
	{
		m_targetNetUpdateFreq = targetNetUpdateFreq;
		return *this;
	}
	DeepActorCriticOptions& multiStep(uint32_t step, bool compound)
	{
		m_multiStep = step;
		m_multiStepCompound = compound;
		return *this;
	}
	DeepActorCriticOptions& experienceReplay(uint32_t replayMemorySize, uint32_t warmUpSize, size_t learnFreq)
	{
		m_experienceReplay = ExperienceReplay::experience_replay;
		m_replayMemorySize = replayMemorySize;
		m_warmUpSize = warmUpSize;
		m_learnFreq = learnFreq;
		return *this;
	}
	DeepActorCriticOptions& prioritizedExperienceReplay(uint32_t replayMemorySize, uint32_t warmUpSize, size_t learnFreq, float prioritizedAlpha = 1.0f, float prioritizedBeta = 1.0f, float prioritizedEpsilon = FLT_EPSILON)
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


//template<typename State_t, typename Action_t>
template<typename PolicyNet_t, typename StateValueNet_t, typename PolicyFunction_t = PolicyNet_t>
class DeepActorCritic : public Agent<typename PolicyNet_t::State_t, typename PolicyNet_t::Action_t>
{
public:
	//typedef State_t State_t;
	//typedef Action_t Action_t;
	//typedef PolicyNet<State_t, Action_t> PolicyNet_t;
	//typedef StateValueNet<State_t> StateValueNet_t;
	//typedef PolicyFunction<State_t, Action_t> PolicyFunction_t;
	typedef typename PolicyNet_t::State_t State_t;
	typedef typename PolicyNet_t::Action_t Action_t;
	typedef paf::SharedPtr<PolicyNet_t> PolicyNetPtr;
	typedef paf::SharedPtr<StateValueNet_t> StateValueNetPtr;
	typedef std::shared_ptr<Optimizer> OptimizerPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef paf::SharedPtr<DeepActorCritic> DeepActorCriticPtr;
public:
	DeepActorCritic(PolicyNetPtr actorNet, StateValueNetPtr criticNet, OptimizerPtr optimizer, PolicyFunctionPtr policy, const DeepActorCriticOptions& options) :
		m_actorNet(actorNet),
		m_criticNet(criticNet),
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
		m_criticTargetNet = StateValueNetPtr::Make(*criticNet.get()->get());

		uint32_t multiStep = options.multiStep();
		if (multiStep > 1)
		{
			m_multiStepBuffer.initialize(multiStep);
		}
		if (m_learnFreq < 1)
		{
			m_learnFreq = 1;
		}
		if(useTargetNet())
		{
			NN_copyParameters(m_criticTargetNet.get()->get(), m_criticNet.get()->get());
		}
		uint32_t bufferCapacity;
		if(ExperienceReplay::no_experience_replay == m_experienceReplay)
		{
			bufferCapacity = m_batchSize + multiStep * 2;
		}
		else
		{
			bufferCapacity = std::max(m_replayMemorySize, m_warmUpSize);
		}
		m_trajectoryBuffer.initialize(bufferCapacity, false, ExperienceReplay::prioritized_experience_replay == m_experienceReplay);
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
		return m_action;
	}

	void lastStep(float reward, const State_t& nextState, bool terminated)
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
			else if(m_trajectoryBuffer.size() < m_batchSize)
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

		Tensor stateTensor = MakeTensor<State_t>(torch::kFloat32, batchSize);
		Tensor actionTensor = MakeTensor<Action_t>(torch::kInt64, batchSize);
		Tensor rewardTensor = MakeTensor<float>(torch::kFloat32, batchSize);
		Tensor nextStateTensor = MakeTensor<State_t>(torch::kFloat32, batchSize);
		Tensor nextDiscountTensor = MakeTensor<float>(torch::kFloat32, batchSize);
		Tensor weightTensor = MakeTensor<float>(torch::kFloat32, batchSize);

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

		Tensor valueTensor = m_criticNet->forward(stateTensor);
		assert(valueTensor.dim() == 2 && valueTensor.size(0) == batchSize && valueTensor.size(1) == 1);

		Tensor nextValueTensor;
		if (useTargetNet())
		{
			torch::NoGradGuard nograd;
			nextValueTensor = m_criticTargetNet->forward(nextStateTensor);
		}
		else
		{
			torch::NoGradGuard nograd;
			nextValueTensor = m_criticNet->forward(nextStateTensor).detach();//semi gradient
		}
		assert(nextValueTensor.dim() == 2 && nextValueTensor.size(0) == batchSize && nextValueTensor.size(1) == 1);

		Tensor targetTensor = rewardTensor + nextValueTensor * nextDiscountTensor;
		assert(targetTensor.dim() == 2 && targetTensor.size(0) == batchSize && targetTensor.size(1) == 1);
		Tensor deltaTensor = valueTensor - targetTensor;//neg
		assert(deltaTensor.dim() == 2 && deltaTensor.size(0) == batchSize && deltaTensor.size(1) == 1);

		Tensor criticLossTensor;
		if (ExperienceReplay::prioritized_experience_replay == m_experienceReplay)
		{
			Tensor deltaTensor = targetTensor - valueTensor;
			assert(deltaTensor.dim() == 2 && deltaTensor.size(0) == m_batchSize && deltaTensor.size(1) == 1);
			m_trajectoryBuffer.updatePriorities(m_sampleIndices, deltaTensor, m_batchSize, m_prioritizedAlpha, m_prioritizedEpsilon);
			Tensor costTensor = torch::nn::functional::mse_loss(valueTensor, targetTensor, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone)) * weightTensor;
			assert(costTensor.dim() == 2 && costTensor.size(0) == m_batchSize && costTensor.size(1) == 1);
			criticLossTensor = torch::mean(costTensor);
		}
		else
		{
			criticLossTensor = torch::mean(torch::nn::functional::mse_loss(valueTensor, targetTensor));
		}

		Tensor logProbTensor = torch::nn::functional::log_softmax(m_actorNet->forward(stateTensor), 1);
		Tensor actorLossTensor = torch::sum(logProbTensor.gather(1, actionTensor) * deltaTensor.detach());

		m_optimizer->zero_grad();
		actorLossTensor.backward();
		criticLossTensor.backward();
		m_optimizer->step();

		if (useTargetNet())
		{
			if (m_learnCount % m_targetNetUpdateFreq == 0)
			{
				NN_copyParameters(m_criticTargetNet->module(), m_criticNet->module());
			}
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
	PolicyNetPtr m_actorNet;
	StateValueNetPtr m_criticNet;
	OptimizerPtr m_optimizer;
	PolicyFunctionPtr m_policy;
	StateValueNetPtr m_criticTargetNet;
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
public:
	static DeepActorCriticPtr Make(PolicyNetPtr actorNet, StateValueNetPtr criticNet, OptimizerPtr optimizer, PolicyFunctionPtr policy, const DeepActorCriticOptions& options)
	{
		return DeepActorCriticPtr::Make(actorNet, criticNet, optimizer, policy, options);
	}
};


//template<typename PolicyNet_t, typename StateValueNet_t, typename Policy_t = PolicyNet_t>
//inline static DeepActorCritic<PolicyNet_t, StateValueNet_t, Policy_t> MakeDeepActorCritic(
//	PolicyNet_t& actorNet, 
//	StateValueNet_t& criticNet, 
//	Optimizer& optimizer, 
//	Policy_t& policy, 
//	const DeepActorCriticOptions& options)
//{
//	return DeepActorCritic(actorNet, criticNet, optimizer, policy, options);
//}

END_RLTL_IMPL