idf_component_register(
    SRCS "first.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp-tflite-micro
    PRIV_REQUIRES esp-tflite-micro
)

target_compile_options(${COMPONENT_LIB} PRIVATE 
    -fno-exceptions 
    -fno-rtti
)

target_link_options(${COMPONENT_LIB} PRIVATE 
    -Wl,--start-group 
    -lstdc++ 
    -lsupc++ 
    -Wl,--end-group
)
