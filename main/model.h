#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

void setup_tflite(void);
void run_inference(const float *input, int length);

#ifdef __cplusplus
}
#endif

#endif  // MODEL_H