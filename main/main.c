#include <stdio.h>

// Include model.h with C++ compatibility
#ifdef __cplusplus
extern "C" {
#endif
#include "model.h"
#ifdef __cplusplus
}
#endif


void app_main(void) {


    setup_tflite();

    float example_input[] = {2233,2213,4000,3.5,77,274,464,999};
    int input_length = sizeof(example_input) / sizeof(example_input[0]);

    run_inference(example_input, input_length);

    // main loop or other logic can go here
    while (1) { }
}