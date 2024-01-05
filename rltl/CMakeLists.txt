set(ProjectName rltl)

set(InterfaceFiles
"utility.i"
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
"impl/array_vector.h"
"impl/array.h"
"impl/callback.h"
"impl/deep_actor_critic.h"
"impl/deep_q_network.h"
"impl/deep_reinforce.h"
"impl/environment.h"
"impl/expected_sarsa.h"
"impl/exploration.h"
"impl/monte_carlo_control.h"
"impl/monte_carlo_prediction.h"
"impl/multi_step_buffer.h"
"impl/neural_network.h"
"impl/num_array.h"
"impl/policy_net.h"
"impl/policy_state_value_net.h"
"impl/q_learning.h"
"impl/random.h"
"impl/replay_buffer.h"
"impl/replay_memory.h"
"impl/sarsa.h"
"impl/space_transform.h"
"impl/space.h"
"impl/state_value_net.h"
"impl/state_value_table.h"
"impl/temporal_difference_prediction.h"
"impl/trainer.h"
"impl/test.cpp"
"impl/trajectory_buffer.h"
"impl/utility.h"
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
