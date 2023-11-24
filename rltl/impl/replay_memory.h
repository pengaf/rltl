#pragma once
#include "utility.h"
#include <vector>
#include "random.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t, typename Reward_t>
class ReplayMemory
{
public:
	ReplayMemory(size_t capacity):
		m_index(0),
		m_size(0)
	{
		if (capacity < 1)
		{
			capacity = 1;
		}
		m_capacity = capacity;
		m_states = new State_t[capacity];
		m_actions = new Action_t[capacity];
		m_rewards = new Reward_t[capacity];
		m_nextStates = new State_t[capacity];
		m_terminateds = new bool[capacity];
	}
	~ReplayMemory()
	{
		delete[]m_terminateds;
		delete[]m_nextStates;
		delete[]m_rewards;
		delete[]m_actions;
		delete[]m_states;
	}
public:
	void store(const State_t& state, const Action_t& action, Reward_t reward, const State_t& nextState, bool terminated)
	{
		size_t index = m_index;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_terminateds[index] = terminated;
		m_index = (index + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
	}
	void sample(State_t& state, Action_t& action, Reward_t& reward, State_t& nextState, bool& terminated) const
	{
		assert(0 < m_size);
		index = Random::randint(m_size);
		state = m_states[index];
		action = m_actions[index];
		reward = m_rewards[index];
		nextState = m_nextStates[index];
		terminated = m_terminateds[index];
	}
	size_t size() const
	{
		return m_size;
	}
public:
	State_t* m_states{};
	Action_t* m_actions{};
	State_t* m_rewards{};
	State_t* m_nextStates{};
	bool* m_terminateds{};
	size_t m_capacity{};
	size_t m_index{};
	size_t m_size{};
};

END_RLTL_IMPL