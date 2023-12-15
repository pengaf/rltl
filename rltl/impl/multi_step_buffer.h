#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
struct MultiStepBuffer
{
public:
	struct SAR
	{
		State_t state;
		Action_t action;
		float accReward;
		float accDiscountRate;
	};
public:
	~MultiStepBuffer()
	{
		delete[] m_buffer;
	}

	void initialize(uint32_t capacity)
	{
		m_capacity = capacity;
		m_buffer = new SAR[capacity];
	}

	operator bool() const
	{
		return m_capacity > 1;
	}

	void reset()
	{
		m_stepCount = 0;
	}

	uint32_t multiStep() const
	{
		return m_capacity;
	}

	uint32_t stepCount() const
	{
		return m_stepCount;
	}

	//bool full() const
	//{
	//	return m_stepCount >= m_capacity;
	//}

	void append(const State_t& state, const Action_t& action, float reward, float discountRate)
	{
		uint32_t index = m_stepCount % m_capacity;
		m_buffer[index].state = state;
		m_buffer[index].action = action;
		m_buffer[index].accReward = reward;
		m_buffer[index].accDiscountRate = discountRate;
		++m_stepCount;

		uint32_t count = m_stepCount < m_capacity ? m_stepCount : m_capacity;
		for (uint32_t i = 1; i < count; ++i)
		{
			uint32_t prev = (index + m_capacity - i) % m_capacity;
			reward *= discountRate;
			m_buffer[prev].accReward += reward;
			m_buffer[prev].accDiscountRate *= discountRate;
		}
	}

	SAR* getHead()
	{
		//assert(m_buffer);
		//assert(full());
		return &m_buffer[m_stepCount % m_capacity];
	}

	SAR* get(uint32_t step)
	{
		assert(m_buffer);
		if (step < m_stepCount && m_stepCount <= step + m_capacity)
		{
			return &m_buffer[step % m_capacity];
		}
		return nullptr;
	}

	uint32_t count() const
	{
		return m_stepCount < m_capacity ? m_stepCount : m_capacity;
	}

	uint32_t headIndex() const
	{
		assert(full());
		return index = m_stepCount % m_capacity;
	}

	uint32_t index(uint32_t step) const
	{
		assert(step < m_stepCount && step + capacity >= m_stepCount);
		return step % capacity;
	}

	void append(const State_t& state, const Action_t& action, float reward)
	{
		uint32_t index = m_stepCount % m_capacity;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		++m_stepCount;
	}

	float headDiscountedReturn(float discountRate)
	{
		assert(full());
		discountedReturn(m_stepCount - m_capacity, discountRate)
	}

	float discountedReturn(uint32_t startStep, float discountRate)
	{
		assert(startStep < m_stepCount);
		uint32_t count = m_stepCount - startStep;
		assert(count <= m_capacity);
		float accReward = 0;
		for (uint32_t i = 0; i < capacity; ++i)
		{
			accReward = accReward * discountRate + m_rewards[(m_stepCount - 1 - i) % capacity];
		}
		return accReward;
	}

public:
	uint32_t m_capacity{};
	uint32_t m_stepCount{};
	SAR* m_buffer{};
};


END_RLTL_IMPL
