#include "value_function.h"
#include "space.h"

namespace rltl
{
	StateValueTable::StateValueTable(DiscreteSpace* space)
	{
		m_start = space->start();
		m_count = space->count();
		m_values.resize(m_count);
	}
}
