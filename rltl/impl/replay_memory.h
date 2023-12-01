#pragma once
#include "utility.h"
#include <vector>
#include "random.h"
#include "neural_network.h"

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
		m_nonterminals = new bool[capacity];
	}
	~ReplayMemory()
	{
		delete[]m_nonterminals;
		delete[]m_nextStates;
		delete[]m_rewards;
		delete[]m_actions;
		delete[]m_states;
	}
public:
	void store(const State_t& state, const Action_t& action, Reward_t reward, const State_t& nextState, bool nonterminal)
	{
		size_t index = m_index;
		m_states[index] = state;
		m_actions[index] = action;
		m_rewards[index] = reward;
		m_nextStates[index] = nextState;
		m_nonterminals[index] = nonterminal;
		m_index = (index + 1) % m_capacity;
		if (m_size < m_capacity)
		{
			++m_size;
		}
	}
	template<typename OtherState_t, typename OtherAction_t, typename OtherReward_t, typename OtherNonterminal_t>
	void sample(OtherState_t& state, OtherAction_t& action, OtherReward_t& reward, OtherState_t& nextState, OtherNonterminal_t& nonterminal) const
	{
		assert(0 < m_size);
		int index = Random::randint(m_size);
		assign(state, m_states[index]);
		assign(action, m_actions[index]);
		assign(reward, m_rewards[index]);
		assign(nextState, m_nextStates[index]);
		assign(nonterminal, m_nonterminals[index]);
	}
	//template<typename OtherStates_t, typename OtherActions_t, typename OtherRewards_t, typename OtherNonterminal_t>
	//void samples(OtherStates_t& states, OtherActions_t& actions, OtherRewards_t& rewards, OtherStates_t& nextStates, OtherNonterminal_t& nonterminals, uint32_t numSamples) const
	//{
	//	for (uint32_t i = 0; i < numSamples; ++i)
	//	{
	//	}
	//	assert(0 < m_size);
	//	index = Random::randint(m_size);
	//	state = m_states[index];
	//	action = m_actions[index];
	//	reward = m_rewards[index];
	//	nextState = m_nextStates[index];
	//	nonterminal = m_nonterminals[index];
	//}
	size_t size() const
	{
		return m_size;
	}
public:
	State_t* m_states{};
	Action_t* m_actions{};
	Reward_t* m_rewards{};
	State_t* m_nextStates{};
	bool* m_nonterminals{};
	size_t m_capacity{};
	size_t m_index{};
	size_t m_size{};
};

END_RLTL_IMPL