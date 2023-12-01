#pragma once
#include "utility.h"
#include "neural_network.h"

BEGIN_RLTL_IMPL

template<typename Net_t, typename State_t, typename Action_t, typename Value_t = float>
class ActionValueNet
{
public:
	typedef Net_t Net_t;
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Value_t Value_t;
public:
	ActionValueTable(Net_t& net) :
		m_net(net)
	{}
public:
	Action_t firstMaxAction(const State_t& state) const
	{
		return NN_firstMaxAction(m_net, state);
	}
	Action_t randomMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
		return firstMaxAction(state);
	}
public:
	Net_t& m_net;
};

END_RLTL_IMPL
