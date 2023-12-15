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

	typedef Agent_t::State_t State_t;
	typedef Agent_t::Action_t Action_t;
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
		if (callback)
		{
			callback->beginStep(numEpisodes, episode, 0);
		}
		State_t state = environment.reset();
		Action_t action = agent.firstStep(state);
		uint32_t numSteps = 0;
		float totalReward = 0;
		while (true)
		{
			float reward;
			State_t nextState;
			EnvironmentStatus envStatus = environment.step(reward, nextState, action);
			++numSteps;
			totalReward += reward;
			if (EnvironmentStatus::es_normal == envStatus)
			{
				action = agent.nextStep(reward, nextState);
				if (callback)
				{
					callback->endStep(numEpisodes, episode, numSteps, reward);
					callback->beginStep(numEpisodes, episode, numSteps + 1);
				}
			}
			else
			{
				agent.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
				if (callback)
				{
					callback->endStep(numEpisodes, episode, numSteps, reward);
				}
				break;
			}
		}
		if (callback)
		{
			callback->endEpisode(numEpisodes, episode, numSteps, totalReward);
		}
	}
	if (callback)
	{
		callback->endTrain(numEpisodes);
	}

}




//template<typename State_t, typename Action_t, typename StateValueFunction_t>
//class MonteCarloPrediction
//{
//public:
//	Action_t firstStep(State_t& state)
//	{
//		m_state = state;
//		return m_agent.takeAction(state);
//	}
//	Action_t nextStep(float reward, const State_t& nextState, bool terminated)
//	{
//		SR sr;
//		sr.state = m_state;
//		sr.reward = reward;
//		m_state = nextState;
//		m_stateRewards.emplace_back(sr);
//	}
//	void endEpisode()
//	{
//		StateValueFunction_t::float g = 0;
//		size_t count = m_stateRewards.size();
//		for (size_t i = 0; i < count; ++i)
//		{
//			SR& sr = m_stateRewards[count - 1 - i];
//			g = g * m_discountRate + sr.reward;
//			StateValueFunction_t::float value = m_valueFunction.getValue(sr.state);
//			StateValueFunction_t::float newValue = value + (g - value) * m_learningRate;
//			m_valueFunction.setValue(sr.state, newValue);
//		}
//	}
//public:
//	struct SR
//	{
//		State_t state;
//		float reward;
//	};
//	float m_discountRate;
//	float m_learningRate;
//	State_t m_state;
//	std::vector<SR> m_stateRewards;
//	StateValueFunction_t m_valueFunction;
//	Agent_t& m_agent;
//};


//template<typename State_t, typename Action_t, typename StateValueFunction_t>
//class TemporalDifferencePrediction
//{
//public:
//	Action_t beginEpisode(State_t& state)
//	{
//		m_state = state;
//		return m_valueFunction.takeAction(state);
//	}
//	Action_t step(float reward, const State_t& nextState, bool terminated)
//	{
//		SR sr;
//		sr.state = m_state;
//		sr.reward = reward;
//		m_state = nextState;
//		m_stateRewards.emplace_back(sr);
//	}
//	void endEpisode()
//	{
//		StateValueFunction_t::float g = 0;
//		size_t count = m_stateRewards.size();
//		for (size_t i = 0; i < count; ++i)
//		{
//			SR& sr = m_stateRewards[count - 1 - i];
//			g = g * m_discountRate + sr.reward;
//			StateValueFunction_t::float value = m_valueFunction.getValue(sr.state);
//			StateValueFunction_t::float newValue = value + (g - value) * m_learningRate;
//			m_valueFunction.setValue(sr.state, newValue);
//		}
//	}
//public:
//	StateValueFunction_t m_valueFunction;
//};



END_RLTL_IMPL

