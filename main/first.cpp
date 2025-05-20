#include <stdio.h>
#include <cstring>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// Model parameters
static constexpr int kNumInputs = 14;  // 14 input features
static constexpr int kNumOutputs = 6;  // 6 output classes
static constexpr int kTensorArenaSize = 32 * 1024;  // 32KB should be sufficient for this model
static uint8_t tensor_arena[kTensorArenaSize];

// Class labels - must match the order in your training script
static const char* kClassLabels[] = {
    "normal_posture",
    "pelvis_rearward_rotation",
    "reclining_back",
    "rounded_back",
    "thorax_forward_rotation",
    "thorax_rearward_rotation"
};

// Test input data - replace with actual sensor data
// These are just placeholder values - you should replace them with real sensor data
static const float test_input[kNumInputs] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // First 6 values
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // Next 6 values
    0.0f, 0.0f                           // Last 2 values (total 14)
};

extern "C" void app_main() {
    // Load the model
    const tflite::Model* model = tflite::GetModel(posture_classification_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version mismatch\n");
        return;
    }

    // Set up the operations resolver - we only need these basic ops
    static tflite::MicroMutableOpResolver<3> resolver;  // Only need 3 ops now
    
    // Add operations used by the model
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        printf("Failed to add FULLY_CONNECTED op\n");
        return;
    }
    
    if (resolver.AddSoftmax() != kTfLiteOk) {
        printf("Failed to add SOFTMAX op\n");
        return;
    }
    
    if (resolver.AddReshape() != kTfLiteOk) {
        printf("Failed to add RESHAPE op\n");
        // Continue anyway, as this might not be used
    }

    // Create the interpreter
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);

    // Allocate tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed\n");
        return;
    }

    // Get input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    // Print tensor information
    printf("Model input dimensions: %d\n", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        printf("  Dim %d: %d\n", i, input->dims->data[i]);
    }
    
    printf("Model output dimensions: %d\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        printf("  Dim %d: %d\n", i, output->dims->data[i]);
    }

    // Copy test input data to the input tensor
    if (input->type == kTfLiteFloat32) {
        printf("Copying float input data...\n");
        float* input_data = input->data.f;
        for (int i = 0; i < kNumInputs; i++) {
            input_data[i] = test_input[i];
            printf("Input[%d] = %f\n", i, input_data[i]);
        }
    } else {
        printf("Unsupported input type\n");
        return;
    }

    // Run inference
    printf("\nRunning inference...\n");
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Inference failed\n");
        return;
    }

    // Process the output
    printf("\nInference results:\n");
    
    float max_prob = 0.0f;
    int max_index = 0;
    
    if (output->type == kTfLiteFloat32) {
        float* output_data = output->data.f;
        
        // Print all class probabilities
        printf("Class probabilities:\n");
        for (int i = 0; i < kNumOutputs; i++) {
            printf("  %s: %.2f%%\n", kClassLabels[i], output_data[i] * 100.0f);
            
            if (output_data[i] > max_prob) {
                max_prob = output_data[i];
                max_index = i;
            }
        }
    } else {
        printf("Unsupported output type\n");
        return;
    }
    
    // Print the final prediction
    printf("\nPredicted class: %s (%.2f%% confidence)\n", 
           kClassLabels[max_index], max_prob * 100.0f);
    
    // Print memory usage
    printf("\nMemory usage:\n");
    printf("  Tensor arena: %d/%d bytes used\n", 
           interpreter.arena_used_bytes(), kTensorArenaSize);
}
