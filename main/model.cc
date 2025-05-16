
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "model_data.h"

constexpr int kTensorArenaSize = 10 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup_tflite() {
    // Map the model from the binary data array
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch! Model was generated with schema version %d, but this runtime expects version %d.",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Register only the operators that your model requires (example: 3 ops)
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();


    // Create the interpreter instance statically
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // Allocate memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    // Get pointers to model input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    MicroPrintf("TFLite Micro setup complete");
}

void run_inference(const float* input_data, int input_length) {
    if (!interpreter || !input || !output) {
        MicroPrintf("Interpreter not initialized");
        return;
    }

    // Copy input data to the input tensor
    for (int i = 0; i < input_length; ++i) {
        input->data.f[i] = input_data[i];
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
    }

    // Output inference result(s)
    MicroPrintf("Inference completed. Output[0]=%f", output->data.f[0]);
    // If your model has more than one output element, print those as well
    // for (int i = 0; i < output->dims->data[0]; ++i) {
    //     MicroPrintf("Output[%d]=%f", i, output->data.f[i]);
    // }
}

// Example usage (uncomment and use in your main application entry point)
/*
void app_main() {
    setup_tflite();
    float example_input[] = {5.1, 3.5, 1.4, 0.2}; // Replace with your real input data
    run_inference(example_input, 4);
}
*/