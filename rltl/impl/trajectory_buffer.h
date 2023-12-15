#pragma once
#include "utility.h"
#include "random.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t, typename Priority_t = float, typename PrioritySum_t = double>
class TrajectoryBuffer
{
public:
	~TrajectoryBuffer()
	{
		delete[]m_prioritySums;
		delete[]m_priorities;
		delete[]m_nextActions;
		delete[]m_nextDiscounts;
		delete[]m_nextStates;
		delete[]m_rewards;
		delete[]m_actions;
		delete[]m_states;
	}
public:
	void initialize(
		uint32_t capacity,
		bool needNextAction,
		bool needPriority)
	{
		if (needPriority)
		{
			size_t alignedCapacity = 2;
			while (alignedCapacity < capacity)
			{
				alignedCapacity *= 2;
			}
			capacity = alignedCapacity;
		}

		m_capacity = capacity;
		m_states = new State_t[capacity];
		m_actions = new Action_t[capacity];
		m_rewards = new float[capacity];
		m_nextStates = new State_t[capacity];
		m_nextDiscounts = new float[capacity];
		if (needNextAction)
		{
			//for sarsa
			m_nextActions = new Action_t[capacity];
		}
		if (needPriority)
		{
			m_priorities = new Priority_t[capacity];
			std::memset(m_priorities, 0, sizeof(Priority_t)*(capacity));
			m_prioritySums = new PrioritySum_t[capacity - 1];
			std::memset(m_prioritySums, 0, sizeof(PrioritySum_t)*(capacity - 1));
		}
	}
public:
	uint32_t size() const
	{
		return m_size;
	}

	uint32_t append(
		const State_t& state, 
		const Action_t& action, 
		float reward, 
		const State_t& nextState, 
		float nextDiscount)
	{
		assert(nullptr == m_nextActions);
		uint32_t index = m_end;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_nextDiscounts[index] = nextDiscount;
		if (m_priorities)
		{
			updatePriority(index, m_maxPriority);
		}
		m_end = (m_end + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
		else
		{
			m_begin = (m_begin + 1) % m_capacity;
		}
		assert((m_begin + m_size) % m_capacity == m_end);
		return index;
	}

	uint32_t append(
		const State_t& state, 
		const Action_t& action, 
		float reward, 
		const State_t& nextState, 
		float nextDiscount, 
		const Action_t& nextAction)
	{
		assert(nullptr != m_nextActions);
		uint32_t index = m_end;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_nextDiscounts[index] = nextDiscount;
		m_nextActions[index] = nextAction;
		if (m_priorities)
		{
			updatePriority(index, m_maxPriority);
		}
		m_end = (m_end + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
		else
		{
			m_begin = (m_begin + 1) % m_capacity;
		}
		assert((m_begin + m_size) % m_capacity == m_end);
		return index;
	}

public:
	//for sequential retrive
	void pop(
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		uint32_t batchSize)
	{
		assert(nullptr == m_priorities);
		assert(nullptr == m_nextActions);
		assert(0 < batchSize && batchSize <= m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = (m_begin + i) % m_capacity;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
		}
		m_begin = (m_begin + batchSize) % m_capacity;
		m_size -= batchSize;
		assert((m_begin + m_size) % m_capacity == m_end);
	}

	void pop(
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		Tensor& nextActionTensor, 
		uint32_t batchSize)
	{
		assert(nullptr == m_priorities);
		assert(nullptr != m_nextActions);
		assert(0 < batchSize && batchSize <= m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		auto nextActions = nextActionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = (m_begin + i) % m_capacity;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
			assign(nextActions[i], m_nextActions[index]);
		}
		m_begin = (m_begin + batchSize) % m_capacity;
		m_size -= batchSize;
		assert((m_begin + m_size) % m_capacity == m_end);
	}

public:
	//for experience replay
	void sample(
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		uint32_t batchSize) const
	{
		assert(nullptr == m_nextActions);
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = Random::randuint(m_size);// (m_begin + Random::randuint(m_size)) % m_capacity;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
		}
	}

	void sample(
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		Tensor& nextActionTensor, 
		uint32_t batchSize) const
	{
		assert(nullptr != m_nextActions);
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		auto nextActions = nextActionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = Random::randuint(m_size);// (m_begin + Random::randuint(m_size)) % m_capacity;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
			assign(nextActions[i], m_nextActions[index]);
		}
	}

public:
	//for prioritized experience replay
	void sample(
		std::vector<uint32_t>& indices, 
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		Tensor& weightTensor, 
		uint32_t batchSize,
		float prioritizedBeta) const
	{
		assert(nullptr == m_nextActions);
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		auto weights = weightTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = sampleIndexSumTree();
			assert(index < m_size);
			indices[i] = index;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
			weights[i][0] = std::pow(m_minPriority / m_priorities[index], prioritizedBeta);
		}
	}

	void sample(
		std::vector<uint32_t>& indices, 
		Tensor& stateTensor, 
		Tensor& actionTensor, 
		Tensor& rewardTensor, 
		Tensor& nextStateTensor, 
		Tensor& nextDiscountTensor, 
		Tensor& nextActionTensor, 
		Tensor& weightTensor, 
		uint32_t batchSize,
		float prioritizedBeta) const
	{
		assert(nullptr != m_nextActions);
		assert(0 < batchSize && 0 < m_size);
		auto states = stateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto actions = actionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto rewards = rewardTensor.accessor<float, 2>();
		auto nextStates = nextStateTensor.accessor<float, GetDimension<State_t>::dim() + 1>();
		auto nextDiscounts = nextDiscountTensor.accessor<float, 2>();
		auto nextActions = nextActionTensor.accessor<int64_t, GetDimension<Action_t>::dim() + 1>();
		auto weights = weightTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			uint32_t index = sampleIndexSumTree();
			assert(index < m_size);
			indices[i] = index;
			assign(states[i], m_states[index]);
			assign(actions[i], m_actions[index]);
			assign(rewards[i], m_rewards[index]);
			assign(nextStates[i], m_nextStates[index]);
			assign(nextDiscounts[i], m_nextDiscounts[index]);
			assign(nextActions[i], m_nextActions[index]);
			weights[i][0] = std::pow(m_minPriority / m_priorities[index], prioritizedBeta);
		}
	}

	void updatePriorities(const std::vector<uint32_t>& indices, const Tensor& deltaTensor, uint32_t batchSize, float prioritizedAlpha, float prioritizedEpsilon)
	{
		assert(indices.size() == batchSize);
		auto deltas = deltaTensor.accessor<float, 2>();
		for (uint32_t i = 0; i < batchSize; ++i)
		{
			size_t index = indices[i];
			assert(index < m_size);
			float delta = std::abs(deltas[i][0]);
			float priority = std::pow(delta + prioritizedEpsilon, prioritizedAlpha);
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
	uint32_t sampleIndexSumTree() const
	{
		Priority_t priority = Random::rand() * m_prioritySums[0];
		uint32_t leftNode = 1;
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
		uint32_t index = m_priorities[leftNode] > priority ? leftNode : leftNode + 1;
		if (index >= m_size)
		{
			index = m_size - 1;
		}
		return index;
	}
protected:
	uint32_t m_capacity{ 0 };
	uint32_t m_size{ 0 };
	uint32_t m_begin{ 0 };
	uint32_t m_end{ 0 };
	State_t* m_states{ nullptr };
	Action_t* m_actions{ nullptr };
	float* m_rewards{ nullptr };
	State_t* m_nextStates{ nullptr };
	float* m_nextDiscounts{ nullptr };
	Action_t* m_nextActions{ nullptr };
	Priority_t* m_priorities{};
	PrioritySum_t* m_prioritySums{};
	Priority_t m_minPriority{ FLT_MAX };
	Priority_t m_maxPriority{ 1.0f };
};

END_RLTL_IMPL
