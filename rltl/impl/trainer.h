#pragma once
#include "utility.h"
#include "callback.h"
#include <vector>

BEGIN_RLTL_IMPL

template<typename StateSpace_t, typename ActionSpace_t>
class Trainer
{
public:
	typedef StateSpace_t StateSpace_t;
	typedef typename StateSpace_t::Element_t State_t;
	typedef ActionSpace_t ActionSpace_t;
	typedef typename ActionSpace_t::Element_t Action_t;
public:
	void addEpisodeCallback(EpisodeCallback* episodeCallback);
	void removeEpisodeCallback(EpisodeCallback* episodeCallback);
	void addStepCallback(StepCallback* stepCallback);
	void removeStepCallback(StepCallback* stepCallback);
public:
	//template<typename Environment_t, typename Agent_t, typename CallBack_t>
	inline void train(Environment_t& environment, Agent_t& agent, uint32_t numEpisodes)
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
			State_t state = environment.reset();
			Action_t action = agent.firstStep(state);
			uint32_t totalStep = 0;
			float totalReward = 0;
			while (true)
			{
				float reward;
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
					agent.lastStep(reward, nextState, envStatus != EnvironmentStatus::es_terminated);
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
//public:
//	std::vector<EpisodeCallback*> m_episodeCallbacks;
//	std::vector<StepCallback*> m_stepCallbacks;
//	Environment* m_environment;
};


END_RLTL_IMPL