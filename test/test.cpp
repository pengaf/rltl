#include <iostream>
#include "../rltl/impl/action_value_table.h"
#include "../rltl/impl/epsilon_greedy.h"
#include "../rltl/impl/sarsa.h"
#include "../rltl/impl/expected_sarsa.h"
#include "../rltl/impl/q_learning.h"
#include "../rltl/impl/deep_q_network.h"
#include "../rltl/impl/deep_reinforce.h"
#include "../rltl/impl/deep_actor_critic.h"

#include "../rltl/impl/action_value_net.h"
#include "../rltl/impl/state_value_net.h"
#include "../rltl/impl/policy_net.h"

#include "../rltl/impl/algorithm.h"
#include "../rltl/impl/callback.h"
#include "env/cliff_walking.h"
#include "env/mountain_car.h"
#include "env/cart_pole.h"
#include <chrono>

template<typename Agent_t>
class RewardStat
{
public:
	typedef typename Agent_t::State_t State_t;
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
	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_totalReward += reward;
		return m_agent.nextStep(reward, nextState);
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
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
			printf("episode:%d,step:%d,reward:%f,time:%f,step/second: %d\n", episode, totalStep, m_rewardStat.back(), seconds, stepPerSecond);
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

class TestStepCallback : public rltl::impl::Callback
{
public:
	virtual void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step) override
	{
		++m_totalStep;
	}
	virtual void endTrain(uint32_t numEpisodes) override
	{
		std::cout << m_totalStep << std::endl;
	}
	uint32_t m_totalStep{ 0 };
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
	//rltl::impl::Callback* stepCallback = new TestStepCallback;
	typedef CartPole Env;
	Env env;

	rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
	rltl::impl::DeepQLearningOptions options(0.98, 500, 64, 5, 5, false);
	//options.multiStep(4);

	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
	auto agent = rltl::impl::MakeDQN(replayMemory, actionValueFunction, optimizer, epsilonGreedy, options);
	RewardStat2 rewardStat;
	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
	rltl::impl::train(env, agent, 1000, &rewardStat);
}

void test_deep_sarsa()
{
	//rltl::impl::Callback* stepCallback = new TestStepCallback;
	typedef CartPole Env;
	Env env;

	rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
	rltl::impl::DeepSarsaOptions options(0.98, 500, 64, 5, 5);

	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
	auto agent = rltl::impl::MakeDeepSarsa(replayMemory, actionValueFunction, optimizer, epsilonGreedy, options);
	RewardStat2 rewardStat;
	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
	rltl::impl::train(env, agent, 1000, &rewardStat);
}

void test_deep_expected_sarsa()
{
	//rltl::impl::Callback* stepCallback = new TestStepCallback;
	typedef CartPole Env;
	Env env;

	rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
	rltl::impl::DeepExpectedSarsaOptions options(0.98, 500, 64, 5, 5);

	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
	auto agent = rltl::impl::MakeDeepExpectedSarsa(replayMemory, actionValueFunction, optimizer, epsilonGreedy, options);
	RewardStat2 rewardStat;
	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
	rltl::impl::train(env, agent, 1000, &rewardStat);
}

void test_reinforce()
{
	//rltl::impl::Callback* stepCallback = new TestStepCallback;
	typedef CartPole Env;
	Env env;
	rltl::impl::MLPPolicyNet<Env::State_t, Env::Action_t> policyFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1);
	torch::optim::Adam optimizer(policyFunction->parameters(), torch::optim::AdamOptions(1e-3));
	rltl::impl::ReinforceOptions options(0.98);

	auto agent = rltl::impl::MakeReinforce(policyFunction, optimizer, policyFunction, options);
	RewardStat2 rewardStat;
	rltl::impl::train(env, agent, 1000, &rewardStat);
}

void test_actor_critic()
{
	typedef CartPole Env;
	Env env;

	rltl::impl::MLPPolicyNet<Env::State_t, Env::Action_t> actorNet(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1);
	torch::optim::Adam actorOptimizer(actorNet->parameters(), torch::optim::AdamOptions(1e-3));

	rltl::impl::MLPStateValueNet<Env::State_t> criticNet(env.stateSpace().dimension(), 128, 1);
	torch::optim::Adam criticOptimizer(criticNet->parameters(), torch::optim::AdamOptions(1e-3));

	rltl::impl::DeepActorCriticOptions options(0.98, 500, 64, 5);

	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));

	auto agent = rltl::impl::MakeDeepActorCritic(replayMemory, actorNet, actorOptimizer, criticNet, criticOptimizer, actorNet, options);
	RewardStat2 rewardStat;
	rltl::impl::train(env, agent, 1000, &rewardStat);
}

int main()
{
	//test_dqn();
	//test_reinforce();
	//test_table();
	try
	{
		test_dqn();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
	test_dqn();
}

//inline static DeepQNetwork<ActionValueNet_t, Policy_t> MakeDeepQNetwork(ActionValueNet_t& valueNet, Optimizer_t& optimizer, Policy_t& policy, const DeepQNetworkOptions& options)
