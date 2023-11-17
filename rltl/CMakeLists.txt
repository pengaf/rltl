set(ProjectName rltl)

set(HeaderFiles
	"data_type.h"
	"space.h"
	"value_function.h"
	"environment.h"
	"model.h"
	"agent.h"
)
source_group("Header Files" FILES ${HeaderFiles})

set(SourceFiles
	"space.cpp"
	"value_function.cpp"
)
source_group("Source Files" FILES ${SourceFiles})

set(impl
"impl/utility.h"
"impl/value_function.h"
"impl/algorithm.h"
"impl/agent.h"
"impl/explorator.h"
)
source_group("impl" FILES ${impl})

set(math
"math/utility.h"
"math/random.h"
)
source_group("math" FILES ${math})


set(AllFiles
    ${HeaderFiles}
    ${SourceFiles}
    ${impl}
    ${math}
)


add_library(${ProjectName} SHARED ${AllFiles})
#target_link_libraries(${ProjectName} "${TORCH_LIBRARIES}")