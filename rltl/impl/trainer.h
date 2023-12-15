#pragma once
#include "utility.h"
#include "callback.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class Trainer
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Agent<State_t, Action_t> Agent_t;
	typedef Environment<State_t, Action_t> Environment_t;
public:
	void addEpisodeCallback(EpisodeCallback* episodeCallback);
	void removeEpisodeCallback(EpisodeCallback* episodeCallback);
	void addStepCallback(StepCallback* stepCallback);
	void removeStepCallback(StepCallback* stepCallback);
public:
	void trainEpisodes(Agent_t* agent, Environment_t* environment, uint32_t numEpisodes, Callback* callback)
	{
		//if (nullptr == agent || nullptr == environment)
		//{
		//	return;
		//}

		if (callback)
		{
			callback->beginTrain();
		}
		for (uint32_t episode = 0; episode < numEpisodes; ++episode)
		{
			if (callback)
			{
				callback->beginEpisode(episode);
			}
			if (callback)
			{
				callback->beginStep(episode, 0);
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
						callback->endStep(episode, numSteps, reward);
						callback->beginStep(episode, numSteps + 1);
					}
				}
				else
				{
					agent.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
					if (callback)
					{
						callback->endStep(episode, numSteps, reward);
						callback->endEpisode(episode, numSteps, totalReward);
					}
					break;
				}
			}
		}
		if (callback)
		{
			callback->endTrain();
		}
	}

	void trainSteps(Agent_t* agent, Environment_t* environment, uint64_t maxSteps, Callback* callback)
	{
		if (nullptr == agent || nullptr == environment || 0 == maxSteps)
		{
			return;
		}
		if (callback)
		{
			callback->beginTrain();
		}
		for (uint32_t episode = 0; episode < numEpisodes; ++episode)
		{
			if (callback)
			{
				callback->beginEpisode(episode);
			}
			if (callback)
			{
				callback->beginStep(episode, 0);
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
						callback->endStep(episode, numSteps, reward);
						callback->beginStep(episode, numSteps + 1);
					}
				}
				else
				{
					agent.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
					if (callback)
					{
						callback->endStep(episode, numSteps, reward);
						callback->endEpisode(episode, numSteps, totalReward);
					}
					break;
				}
			}
		}
		if (callback)
		{
			callback->endTrain();
		}
	}

};


END_RLTL_IMPL