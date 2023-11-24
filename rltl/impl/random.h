#include "utility.h"
#include <stdlib.h>
//#include <math.h>

BEGIN_RLTL_IMPL

class Random
{
public:
	static float rand()
	{
		return ::rand() / float(RAND_MAX);
	}
	static int randint(int high)
	{
		return ::rand() % high;
	}
};


END_RLTL_IMPL
