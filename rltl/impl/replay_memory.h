#pragma once
#include "utility.h"
#include "../arg.h"
#include <cmath>
#include "random.h"
#include "neural_network.h"
#include "callback.h"

BEGIN_RLTL_IMPL

struct ReplayMemoryOptions
{
	ReplayMemoryOptions(uint32_t capacity) :
		m_capacity(capacity),
		m_prioritized(false),
		m_prioritizedAlpha(0),
		m_prioritizedBeta(0),
		m_prioritizedEpsilon(0)
	{}

	ReplayMemoryOptions(uint32_t capacity, float prioritizedAlpha, float prioritizedBeta, float prioritizedEpsilon = FLT_EPSILON) :
		m_capacity(capacity),
		m_prioritized(true),
		m_prioritizedAlpha(prioritizedAlpha),
		m_prioritizedBeta(prioritizedBeta),
		m_prioritizedEpsilon(prioritizedEpsilon)
	{}
	RLTL_ARG(uint32_t, capacity);
	RLTL_ARG(bool, prioritized);
	RLTL_ARG(float, prioritizedEpsilon);
	RLTL_ARG(float, prioritizedAlpha);
	RLTL_ARG(float, prioritizedBeta);
};


template<typename ReplayMemory_t>
class PrioritizedBetaGrow : public Callback
{
public:
	PrioritizedBetaGrow(ReplayMemory_t& replayMemory, uint32_t numSteps, float initBeta, float finalBeta = 1.0f):
		m_replayMemory(replayMemory),
		m_numSteps(numSteps),
		m_currentStep(0),
		m_initBeta(initBeta),
		m_finalBeta(finalBeta)
	{}
public:
	void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step) override	
	{
		float beta = m_finalBeta;
		if (m_currentStep < m_numSteps)
		{
			beta = m_initBeta + (m_finalBeta - m_initBeta) * double(m_currentStep) / double(m_numSteps);
			++m_currentStep;
		}
		m_replayMemory.prioritizedBeta(beta);
	}
protected:
	ReplayMemory_t& m_replayMemory;
	uint32_t m_numSteps;
	uint32_t m_currentStep;
	float m_initBeta;
	float m_finalBeta;
};

template<typename State_t, typename Action_t, typename Priority_t = float, typename PrioritySum_t = double>
class ReplayMemory
{
public:
	ReplayMemory(const ReplayMemoryOptions& options) :
		m_index(0),
		m_size(0),
		m_prioritizedAlpha(options.prioritizedAlpha()),
		m_prioritizedBeta(options.prioritizedBeta()),
		m_prioritizedEpsilon(options.prioritizedEpsilon())
	{
		size_t capacity = 2;
		while (capacity < options.capacity())
		{
			capacity *= 2;
		}
		m_capacity = capacity;
		m_states = new State_t[capacity];
		m_actions = new Action_t[capacity];
		m_rewards = new float[capacity];
		m_nextStates = new State_t[capacity];
		m_nonterminals = new bool[capacity];
		if (options.prioritized())
		{
			m_priorities = new Priority_t[capacity];
			std::memset(m_priorities, 0, sizeof(Priority_t)*(capacity));
			m_prioritySums = new PrioritySum_t[capacity - 1];
			std::memset(m_prioritySums, 0, sizeof(PrioritySum_t)*(capacity-1));
		}
	}

	~ReplayMemory()
	{
		delete[]m_nonterminals;
		delete[]m_nextStates;
		delete[]m_rewards;
		delete[]m_actions;
		delete[]m_states;
		delete[]m_nextActions;
		delete[]m_priorities;
		delete[]m_prioritySums;
	}
public:
	size_t size() const
	{
		return m_size;
	}

	bool isPrioritized() const
	{
		return m_priorities != nullptr;
	}

	float prioritizedBeta() const
	{
		return m_prioritizedBeta;
	}

	void prioritizedBeta(float beta)
	{
		m_prioritizedBeta = beta;
	}

	//for sarsa
	void prepareNextActions()
	{
		if (!m_nextActions)
		{
			m_nextActions = new Action_t[m_capacity];
		}
	}

	void store(const State_t& state, const Action_t& action, float reward, const State_t& nextState, bool nonterminal)
	{
		size_t index = m_index;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_nonterminals[index] = nonterminal;
		if (m_priorities)
		{
			updatePriority(index, m_maxPriority);
		}
		m_index = (index + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
	}

	void store(const State_t& state, const Action_t& action, float reward, const State_t& nextState, const Action_t& nextAction, bool nonterminal)
	{
		size_t index = m_index;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_nextActions[index] = nextAction;
		m_nonterminals[index] = nonterminal;
		if (m_priorities)
		{
			updatePriority(index, m_maxPriority);
		}
		m_index = (index + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
	}

	////sequential sample and erase 
	//bool popup(Tensor& stateTensor, Tensor& actionTensor, Tensor& rewardTensor, Tensor& nextStateTensor, Tensor& nonterminalTensor, uint32_t batchSize)
	//{
	//	assert(0 < batchSize && batchSize <= m_size);
	//	auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	//	auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
	//	auto rewards = rewardTensor.accessor<float, 2>();
	//	auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
	//	auto nonterminals = nonterminalTensor.accessor<float, 2>();
	//	for (uint32_t i = 0; i < batchSize; ++i)
	//	{
	//		size_t index = (size_t)Random::randint(m_size);
	//		assert(index < m_size);
	//		assign(states[i], m_states[index]);
	//		assign(actions[i], m_actions[index]);
	//		assign(rewards[i], m_rewards[index]);
	//		assign(nextStates[i], m_nextStates[index]);
	//		assign(nonterminals[i], m_nonterminals[index]);
	//	}
	//}

	//random sample
	void sample(Tensor& stateTensor, Tensor& actionTensor, Tensor& rewardTensor, Tensor& nextStateTensor, Tensor& nonterminalTensor, uint32_t batchSize) const
	{
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nonterminals = nonterminalTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = (size_t)Random::randint(m_size);
			assert(index < m_size);
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nonterminals[i], m_nonterminals[index]);
		}
	}

	void sample(Tensor& stateTensor, Tensor& actionTensor, Tensor& rewardTensor, Tensor& nextStateTensor, Tensor& nextActionTensor, Tensor& nonterminalTensor, uint32_t batchSize) const
	{
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextActions = nextActionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto nonterminals = nonterminalTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = (size_t)Random::randint(m_size);
			assert(index < m_size);
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextActions[i], m_nextActions[index]);
			assign(nonterminals[i], m_nonterminals[index]);
		}
	}

	//random sample with priority
	void sample(std::vector<uint32_t>& indices, Tensor& stateTensor, Tensor& actionTensor, Tensor& rewardTensor, Tensor& nextStateTensor, Tensor& nonterminalTensor, Tensor& weightTensor, uint32_t batchSize) const
	{
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nonterminals = nonterminalTensor.accessor<float, 2>();
		auto weights = weightTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = sampleIndexSumTree();
			assert(index < m_size);
			indices[i] = index;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nonterminals[i], m_nonterminals[index]);
			weights[i][0] = std::pow(m_minPriority / m_priorities[index], m_prioritizedBeta);
		}
	}

	void sample(std::vector<uint32_t>& indices, Tensor& stateTensor, Tensor& actionTensor, Tensor& rewardTensor, Tensor& nextStateTensor, Tensor& nextActionTensor, Tensor& nonterminalTensor, Tensor& weightTensor, uint32_t batchSize) const
	{
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextActions = nextActionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto nonterminals = nonterminalTensor.accessor<float, 2>();
		auto weights = weightTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = sampleIndexSumTree();
			assert(index < m_size);
			indices[i] = index;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextActions[i], m_nextActions[index]);
			assign(nonterminals[i], m_nonterminals[index]);
			weights[i][0] = std::pow(m_minPriority / m_priorities[index], m_prioritizedBeta);
		}
	}

	void updatePriorities(const std::vector<uint32_t>& indices, const Tensor& deltaTensor, uint32_t batchSize)
	{
		assert(indices.size() == batchSize);
		auto deltas = deltaTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = indices[i];
			assert(index < m_size);
			float delta = std::abs(deltas[i][0]);
			float priority = std::pow(delta + m_prioritizedEpsilon, m_prioritizedAlpha);
			updatePriority(index, priority);
		}
	}
protected:
	void updatePriority(size_t index, float priority)
	{
		m_priorities[index] = priority;
		if (m_prioritySums)
		{
			updateSumTree(index);
		}
		if (m_minPriority > priority)
		{
			m_minPriority = priority;
		}
		if (m_maxPriority < priority)
		{
			m_maxPriority = priority;
		}
	}
	void updateSumTree(size_t index)
	{
		size_t parent = (index + m_capacity) / 2 - 1;
		m_prioritySums[parent] = m_priorities[index] + m_priorities[index ^ 1];
		while (parent)
		{
			parent = (parent - 1) / 2;
			m_prioritySums[parent] = m_prioritySums[parent * 2 + 1] + m_prioritySums[parent * 2 + 2];
			assert(m_prioritySums[parent] >= m_prioritySums[parent * 2 + 1] && m_prioritySums[parent] >= m_prioritySums[parent * 2 + 2]);
		}
	}
	size_t sampleIndexSumTree() const
	{
		Priority_t priority = Random::rand() * m_prioritySums[0];
		size_t leftNode = 1;
		while (leftNode < m_capacity - 1)
		{
			if (m_prioritySums[leftNode] > priority)
			{
				leftNode = leftNode * 2 + 1;
			}
			else
			{
				priority -= m_prioritySums[leftNode];
				leftNode = (leftNode + 1) * 2 + 1;
			}
		}
		leftNode -= (m_capacity - 1);
		size_t index = m_priorities[leftNode] > priority ? leftNode : leftNode + 1;
		if (index >= m_size)
		{
			index = m_size - 1;
		}
		return index;
	}
protected:
	State_t* m_states{};
	Action_t* m_actions{};
	float* m_rewards{};
	State_t* m_nextStates{};
	Action_t* m_nextActions{};
	bool* m_nonterminals{};
	Priority_t* m_priorities{};
	PrioritySum_t* m_prioritySums{};
	Priority_t m_minPriority{ FLT_MAX };
	Priority_t m_maxPriority{ 1.0f };
	size_t m_capacity{};
	size_t m_index{};
	size_t m_size{};
	float m_prioritizedAlpha;
	float m_prioritizedBeta;
	float m_prioritizedEpsilon;
};

END_RLTL_IMPL