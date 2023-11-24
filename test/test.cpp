#include <iostream>
#include "../rltl/impl/action_value_table.h"
#include "../rltl/impl/epsilon_greedy.h"
#include "../rltl/impl/sarsa.h"
#include "../rltl/impl/expected_sarsa.h"
#include "../rltl/impl/q_learning.h"
#include "../rltl/impl/algorithm.h"
#include "env/cliff_walking.h"
#include "env/mountain_car.h"

template<typename Agent_t>
class RewardStat
{
public:
	typedef typename Agent_t::State_t State_t;
	typedef typename Agent_t::Reward_t Reward_t;
	typedef typename Agent_t::Action_t Action_t;
public:
	RewardStat(Agent_t& agent) :
		m_agent(agent)
	{}
	Action_t firstStep(State_t& firstState)
	{
		m_totalReward = 0;
		return m_agent.firstStep(firstState);
	}
	Action_t nextStep(const Reward_t& reward, const State_t& nextState)
	{
		m_totalReward += reward;
		return m_agent.nextStep(reward, nextState);
	}
	void lastStep(const Reward_t& reward, const State_t& nextState, bool terminated)
	{
		m_totalReward += reward;
		m_rewardStat.push_back(m_totalReward);
		return m_agent.lastStep(reward, nextState, terminated);
	}
public:
	Agent_t& m_agent;
	float m_totalReward;
	std::vector<float> m_rewardStat;
};

class RewardStat2
{
public:
	void beginTrain(uint32_t numEpisodes) 
	{
		m_rewardStat.clear();
		m_rewardStat.reserve(numEpisodes);
		m_maxReward = -FLT_MAX;
	}
	void beginEpisode(uint32_t numEpisodes, uint32_t episode) {}
	void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{
		m_rewardStat.push_back(totalReward);
		if(m_maxReward < totalReward)
		{
			m_maxReward = totalReward;
			printf("episode:%d, reward:%f\n", episode, totalReward);
		}
	}
	void endTrain(uint32_t numEpisodes) {}
public:
	std::vector<float> m_rewardStat;
	float m_maxReward;
};

int main()
{
	//CliffWalking2 env2;
	//rltl::impl::DiscreteEnvironment<CliffWalking2> env(env2);
	//CliffWalking env;
	MountainCar env2;
	rltl::impl::DiscreteEnvironment<MountainCar> env(env2,10);
	
	rltl::impl::ActionValueTable<> actionValueTable(env.stateSpace().count(), env.actionSpace().count());
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueTable, 0.1f);	
	//auto agent = rltl::impl::MakeSarsa(actionValueTable, epsilonGreedy, 0.1f);
	auto agent = rltl::impl::MakeExpectedSarsa(actionValueTable, epsilonGreedy, 0.1f);
	//auto agent = rltl::impl::MakeQLearning(actionValueTable, epsilonGreedy, 0.1f);
	//RewardStat<decltype(agent)> rewardStat(agent);
	RewardStat2 callback;
	rltl::impl::train(env, agent, 5000, &callback);

	//for(float r : callback.m_rewardStat)
	//{
	//	std::cout << r << std::endl;
	//}
}

