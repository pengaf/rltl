#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

class EpisodeCallback
{
public:
	virtual void beginTrain(uint32_t numEpisodes)
	{}
	virtual void beginEpisode(uint32_t numEpisodes, uint32_t episode)
	{}
	virtual void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{}
	virtual void endTrain(uint32_t numEpisodes)
	{}
};

class Callback : public paf::Introspectable
{
public:
	virtual void beginTrain(uint32_t numEpisodes)
	{}
	virtual void beginEpisode(uint32_t numEpisodes, uint32_t episode)
	{}
	virtual void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step)
	{}
	virtual void endStep(uint32_t numEpisodes, uint32_t episode, uint32_t step, float reward)
	{}
	virtual void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{}
	virtual void endTrain(uint32_t numEpisodes)
	{}
};

template<typename CallBack_t, typename... Others>
class CompositeCallBack
{
public:
	CompositeCallBack(CallBack_t& callback, Others... others):
		m_callback(callback)
	{}
public:
	void beginTrain(uint32_t numEpisodes)
	{
		m_callback.beginTrain(numEpisodes);
	}
	void beginEpisode(uint32_t numEpisodes, uint32_t episode)
	{
		m_callback.beginEpisode(numEpisodes, episode);
	}
	void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step)
	{
		m_callback.beginStep(numEpisodes, episode, step);
	}
	void endStep(uint32_t numEpisodes, uint32_t episode, uint32_t step, float reward)
	{
		m_callback.endStep(numEpisodes, episode, step, reward);
	}
	void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{
		m_callback.endEpisode(numEpisodes, episode, totalStep, totalReward);
	}
	void endTrain(uint32_t numEpisodes)
	{
		m_callback.endTrain(numEpisodes);
	}
public:
	CallBack_t& m_callback;
};

template<typename CallBack_t, typename CallBack1_t, typename... Others>
class CompositeCallBack<CallBack_t, CallBack1_t, Others...>
{
public:
	CompositeCallBack(CallBack_t& callback, CallBack1_t& nextCallback, Others... others) :
		m_callback(callback),
		m_otherCallBacks(nextCallback, others...)
	{}
public:
	void beginTrain(uint32_t numEpisodes)
	{
		m_callback.beginTrain(numEpisodes);
		m_otherCallBacks.beginTrain(numEpisodes);
	}
	void beginEpisode(uint32_t numEpisodes, uint32_t episode)
	{
		m_callback.beginEpisode(numEpisodes, episode);
		m_otherCallBacks.beginEpisode(numEpisodes, episode);
	}
	void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step)
	{
		m_callback.beginStep(numEpisodes, episode, step);
		m_otherCallBacks.beginStep(numEpisodes, episode, step);
	}
	void endStep(uint32_t numEpisodes, uint32_t episode, uint32_t step, float reward)
	{
		m_callback.endStep(numEpisodes, episode, step, reward);
		m_otherCallBacks.endStep(numEpisodes, episode, step, reward);
	}
	void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{
		m_callback.endEpisode(numEpisodes, episode, totalStep, totalReward);
		m_otherCallBacks.endEpisode(numEpisodes, episode, totalStep, totalReward);
	}
	void endTrain(uint32_t numEpisodes)
	{
		m_callback.endTrain(numEpisodes);
		m_otherCallBacks.endTrain(numEpisodes);
	}
public:
	CallBack_t& m_callback;
	CompositeCallBack<CallBack1_t, Others...> m_otherCallBacks;
};


template<typename EpsilonGreedy_t>
class DecayEpsilonCallBack : public Callback
{
public:
	DecayEpsilonCallBack(EpsilonGreedy_t& epsilonGreedy, float startEpsilon, float endEpsilon, uint32_t decayEpisodes, uint32_t startDecayEpisode = 0) :
		m_epsilonGreedy(epsilonGreedy),
		m_startEpsilon(startEpsilon),
		m_endEpsilon(endEpsilon),
		m_decayEpisodes(decayEpisodes),
		m_startDecayEpisode(startDecayEpisode)
	{
		if (m_decayEpisodes < 1)
		{
			m_decayEpisodes = 1;
		}
	}
	DecayEpsilonCallBack(EpsilonGreedy_t& epsilonGreedy, float endEpsilon, uint32_t decaySteps, uint32_t startDecayEpisode = 1) :
		CompositeCallBack(epsilonGreedy, epsilonGreedy.getEpsilon(), endEpsilon, decaySteps, startDecayEpisode)
	{}
public:
	void beginEpisode(uint32_t numEpisodes, uint32_t episode)
	{
		float epsilon;
		if (episode < m_startDecayEpisode)
		{
			epsilon = m_startEpsilon;
		}
		else if (episode <= m_startDecayEpisode + m_decayEpisodes)
		{
			epsilon = m_startEpsilon + (m_endEpsilon - m_startEpsilon)*(episode - m_startDecayEpisode) / (m_decayEpisodes);
		}
		else
		{
			epsilon = m_endEpsilon;
		}
		m_epsilonGreedy.setEpsilon(epsilon);
	}
public:
	EpsilonGreedy_t& m_epsilonGreedy;
	float m_startEpsilon;
	float m_endEpsilon;
	uint32_t m_decayEpisodes;
	uint32_t m_startDecayEpisode;
};

END_RLTL_IMPL
