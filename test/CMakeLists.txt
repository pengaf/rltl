set(ProjectName test)

set(HeaderFiles
	"test.h"
)
source_group("Header Files" FILES ${HeaderFiles})

set(SourceFiles
	"test.cpp"
)
source_group("Source Files" FILES ${SourceFiles})

set(env
"env/cart_pole.h"
"env/cart_pole.cpp"
"env/cliff_walking.h"
"env/cliff_walking.cpp"
"env/mountain_car.h"
"env/mountain_car.cpp"
)
source_group("env" FILES ${env})


set(AllFiles
    ${HeaderFiles}
    ${SourceFiles}
    ${env}
)

add_executable(${ProjectName} ${AllFiles})

#add_library(${ProjectName} SHARED ${AllFiles})
set(libpostfix $<$<CONFIG:Debug>:_d>)

target_link_libraries(${ProjectName} "${TORCH_LIBRARIES}")
target_link_libraries(${ProjectName} "G:/pengaf/paf/bin/Release/pafcore${libpostfix}.lib")

message(${PAFCORE_DIR})