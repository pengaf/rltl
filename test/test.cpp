#include <iostream>
#include "../rltl/impl/action_value_table.h"
#include "../rltl/impl/epsilon_greedy.h"
#include "../rltl/impl/sarsa.h"
#include "../rltl/impl/expected_sarsa.h"
#include "../rltl/impl/q_learning.h"
#include "../rltl/impl/deep_q_network.h"
#include "../rltl/impl/mlp_q_net.h"
#include "../rltl/impl/algorithm.h"
#include "../rltl/impl/callback.h"
#include "env/cliff_walking.h"
#include "env/mountain_car.h"
#include <chrono>

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
	Action_t nextStep(Reward_t reward, const State_t& nextState)
	{
		m_totalReward += reward;
		return m_agent.nextStep(reward, nextState);
	}
	void lastStep(Reward_t reward, const State_t& nextState, bool nonterminal)
	{
		m_totalReward += reward;
		m_rewardStat.push_back(m_totalReward);
		return m_agent.lastStep(reward, nextState, nonterminal);
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
		m_allSteps = 0;
		m_start = std::chrono::high_resolution_clock::now();
	}
	void beginEpisode(uint32_t numEpisodes, uint32_t episode) 
	{}
	void endEpisode(uint32_t numEpisodes, uint32_t episode, uint32_t totalStep, float totalReward)
	{
		m_allSteps += totalStep;
		m_rewardStat.push_back(totalReward);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration duration = end - m_start;
		float seconds = duration.count()*0.000000001;
		uint32_t stepPerSecond = m_allSteps / seconds;
		bool newMaxReward = m_maxReward < totalReward;
		if(newMaxReward)
		{
			m_maxReward = totalReward;
		}
		if (newMaxReward || episode * 10 % numEpisodes == 0)
		{
			printf("episode:%d,reward:%f,time:%f,step/second: %d\n", episode, m_rewardStat.back(), seconds, stepPerSecond);
		}
	}
	void endTrain(uint32_t numEpisodes) 
	{
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration duration = end - m_start;
		float seconds = duration.count()*0.000000001;
		uint32_t stepPerSecond = m_allSteps / seconds;
		printf("last:%f, max:%f, time:%f, step/second: %d\n", m_rewardStat.back(), m_maxReward, seconds, stepPerSecond);
	}
public:
	std::vector<float> m_rewardStat;
	float m_maxReward;
	std::chrono::high_resolution_clock::time_point m_start;
	uint32_t m_allSteps;
};

void test_table()
{
	//CliffWalking2 env2;
	//rltl::impl::DiscreteEnvironment<CliffWalking2> env(env2);
	//CliffWalking env;
	MountainCar env2; rltl::impl::DiscreteEnvironment<MountainCar> env(env2,10);


	rltl::impl::ActionValueTable<> actionValueTable(env.stateSpace().count(), env.actionSpace().count());
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueTable, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);

	//auto agent = rltl::impl::MakeSarsa(actionValueTable, epsilonGreedy, 0.1f);
	//auto agent = rltl::impl::MakeExpectedSarsa(actionValueTable, epsilonGreedy, 0.1f);
	auto agent = rltl::impl::MakeQLearning(actionValueTable, epsilonGreedy, 0.1f);
	//RewardStat<decltype(agent)> rewardStat(agent);

	RewardStat2 callback;
	rltl::impl::train(env, agent, 5000, &callback);
}

void test_dqn()
{
	uint32_t numEpisodes = 1000;
	MountainCar env;
	rltl::impl::MLPQNet<MountainCar::State_t, MountainCar::Action_t, float> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
	rltl::impl::DeepQNetworkOptions options(0.98, 10000, 500, 64, 5, 5, false);
	auto agent = rltl::impl::MakeDeepQNetwork(actionValueFunction, optimizer, epsilonGreedy, options);
	RewardStat2 rewardStat;
	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
	rltl::impl::train(env, agent,1000, &rewardStat);
}

int main()
{
	//test_table();
	test_dqn();
}

//inline static DeepQNetwork<ActionValueNet_t, Policy_t> MakeDeepQNetwork(ActionValueNet_t& valueNet, Optimizer_t& optimizer, Policy_t& policy, const DeepQNetworkOptions& options)
