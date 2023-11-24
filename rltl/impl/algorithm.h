#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

//concept
//class TrainCallback
//{
//public:
//	void beginTrain(uint32_t numEpisodes);
//	void beginEpisode(uint32_t numEpisodes, uint32_t episode);
//	void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward);
//	void endTrain(uint32_t numEpisodes);
//};


template<typename Environment_t, typename Agent_t, typename CallBack_t>
inline void train(Environment_t& environment, Agent_t& agent, uint32_t numEpisodes, CallBack_t* callback = nullptr)
{
	static_assert(std::is_same_v<Environment_t::State_t, Agent_t::State_t>);
	static_assert(std::is_same_v<Environment_t::Action_t, Agent_t::Action_t>);
	static_assert(std::is_same_v<Environment_t::Reward_t, Agent_t::Reward_t>);

	typedef Agent_t::State_t State_t;
	typedef Agent_t::Action_t Action_t;
	typedef Agent_t::Reward_t Reward_t;
	if (callback)
	{
		callback->beginTrain(numEpisodes);
	}
	for (uint32_t episode = 0; episode < numEpisodes; ++episode)
	{
		if (callback)
		{
			callback->beginEpisode(numEpisodes, episode);
		}
		State_t state = environment.reset();
		Action_t action = agent.firstStep(state);
		uint32_t totalStep = 0;
		float totalReward = 0;
		while (true)
		{
			Reward_t reward;
			State_t nextState;
			EnvironmentStatus envStatus = environment.step(reward, nextState, action);
			++totalStep;
			totalReward += reward;
			if (EnvironmentStatus::es_normal == envStatus)
			{
				action = agent.nextStep(reward, nextState);
			}
			else
			{
				agent.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
				if (callback)
				{
					callback->endEpisode(numEpisodes, episode, totalStep, totalReward);
				}
				break;
			}
		}
	}
	if (callback)
	{
		callback->endTrain(numEpisodes);
	}

}

template<typename Environment_t, typename Agent_t>
class Train
{
public:
	void train(Environment_t& environment, Agent_t& agent, int32_t numEpisodes)
	{
		typedef Environment_t::State_t State_t;
		typedef Environment_t::Action_t Action_t;
		typedef Environment_t::Reward_t Reward_t;
		for (int32_t episode = 0; episode < numEpisodes; ++episode)
		{
			State_t state = environment.reset();
			Action_t action = agent.firstStep(state);
			while (true)
			{
				Reward_t reward;
				State_t nextState;
				EnvironmentStatus envStatus = environment.step(reward, nextState, action);
				if (EnvironmentStatus::es_normal == envStatus)
				{
					action = agent.nextStep(reward, nextState);
				}
				else
				{
					agent.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
					break;
				}
			}
		}
	}
};



//template<typename State_t, typename Action_t, typename Reward_t, typename StateValueFunction_t>
//class MonteCarloPrediction
//{
//public:
//	Action_t firstStep(State_t& state)
//	{
//		m_state = state;
//		return m_agent.takeAction(state);
//	}
//	Action_t nextStep(const Reward_t& reward, const State_t& nextState, bool terminated)
//	{
//		SR sr;
//		sr.state = m_state;
//		sr.reward = reward;
//		m_state = nextState;
//		m_stateRewards.emplace_back(sr);
//	}
//	void endEpisode()
//	{
//		StateValueFunction_t::Value_t g = 0;
//		size_t count = m_stateRewards.size();
//		for (size_t i = 0; i < count; ++i)
//		{
//			SR& sr = m_stateRewards[count - 1 - i];
//			g = g * m_discountRate + sr.reward;
//			StateValueFunction_t::Value_t value = m_valueFunction.getValue(sr.state);
//			StateValueFunction_t::Value_t newValue = value + (g - value) * m_learningRate;
//			m_valueFunction.setValue(sr.state, newValue);
//		}
//	}
//public:
//	struct SR
//	{
//		State_t state;
//		Reward_t reward;
//	};
//	float m_discountRate;
//	float m_learningRate;
//	State_t m_state;
//	std::vector<SR> m_stateRewards;
//	StateValueFunction_t m_valueFunction;
//	Agent_t& m_agent;
//};


//template<typename State_t, typename Action_t, typename Reward_t, typename StateValueFunction_t>
//class TemporalDifferencePrediction
//{
//public:
//	Action_t beginEpisode(State_t& state)
//	{
//		m_state = state;
//		return m_valueFunction.takeAction(state);
//	}
//	Action_t step(const Reward_t& reward, const State_t& nextState, bool terminated)
//	{
//		SR sr;
//		sr.state = m_state;
//		sr.reward = reward;
//		m_state = nextState;
//		m_stateRewards.emplace_back(sr);
//	}
//	void endEpisode()
//	{
//		StateValueFunction_t::Value_t g = 0;
//		size_t count = m_stateRewards.size();
//		for (size_t i = 0; i < count; ++i)
//		{
//			SR& sr = m_stateRewards[count - 1 - i];
//			g = g * m_discountRate + sr.reward;
//			StateValueFunction_t::Value_t value = m_valueFunction.getValue(sr.state);
//			StateValueFunction_t::Value_t newValue = value + (g - value) * m_learningRate;
//			m_valueFunction.setValue(sr.state, newValue);
//		}
//	}
//public:
//	StateValueFunction_t m_valueFunction;
//};



END_RLTL_IMPL

