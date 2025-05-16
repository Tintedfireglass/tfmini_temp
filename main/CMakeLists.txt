
idf_component_register(SRCS "main.c" "model.cc"
                      INCLUDE_DIRS ".")

# Set C++ standard to C++11 for C++ files
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_11)

# Set compiler flags for C++ files only
set_target_properties(${COMPONENT_LIB} PROPERTIES
    COMPILE_FLAGS "-fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit"
)

# Add C++ specific flags only for C++ files
set_source_files_properties(model.cc PROPERTIES
    COMPILE_FLAGS "-fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit"
)