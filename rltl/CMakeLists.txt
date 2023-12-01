set(ProjectName rltl)

set(InterfaceFiles
	"value_function.i"
)
source_group("Interface Files" FILES ${InterfaceFiles})

set(HeaderFiles
	"data_type.h"
	"space.h"
	"value_function.h"
	"environment.h"
	"model.h"
	"agent.h"
	"arg.h"
)
source_group("Header Files" FILES ${HeaderFiles})

set(SourceFiles
	"space.cpp"
	"value_function.cpp"
)
source_group("Source Files" FILES ${SourceFiles})

set(impl
"impl/action_value_net.h"
"impl/action_value_table.h"
"impl/agent.h"
"impl/algorithm.h"
"impl/array.h"
"impl/callback.h"
"impl/deep_q_network.h"
"impl/environment.h"
"impl/epsilon_greedy.h"
"impl/expected_sarsa.h"
"impl/mlp_q_net.h"
"impl/monte_carlo_control.h"
"impl/monte_carlo_prediction.h"
"impl/neural_network.h"
"impl/q_learning.h"
"impl/random.h"
"impl/reinforce.h"
"impl/replay_memory.h"
"impl/sarsa.h"
"impl/space_normalizer.h"
"impl/space.h"
"impl/temporal_difference_prediction.h"
"impl/utility.h"
"impl/value_function.h"
)
source_group("impl" FILES ${impl})

set(math
"math/utility.h"
"math/random.h"
)
source_group("math" FILES ${math})

set(AllFiles
	${InterfaceFiles}
    ${HeaderFiles}
    ${SourceFiles}
    ${impl}
    ${math}
)

foreach(InterfaceFile IN LISTS InterfaceFiles)
    string(REGEX REPLACE "\\.i$" ".h" HeaderFile ${InterfaceFile})
    #message(${HeaderFile})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${HeaderFile}
	    COMMAND ${IDLCPP_PATH} -emPAFCORE_EXPORT "%(FullPath)"
	    MAIN_DEPENDENCY ${InterfaceFile})
endforeach()

add_library(${ProjectName} SHARED ${AllFiles})
target_link_libraries(${ProjectName} "${TORCH_LIBRARIES}")