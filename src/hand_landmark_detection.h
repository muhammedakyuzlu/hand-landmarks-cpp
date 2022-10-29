#pragma once

#include <vector>
#include <tensorflow/lite/kernels/register.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "typedef_struct.h"

class HandLandmark
{
public:
    // Take a model path as string
    void loadModel(const std::string &hand_model_path);

    void run(cv::Mat frame,hand_landmark_result_t  &hand_result,palm_detection_result_t &palm_result);
    
    // thresh hold
    float confThreshold = 0.5;
    // number of threads
    int nthreads = 4;

    // parameters of original image
    int _img_height;
    int _img_width;

private:  

    // hand model's
    std::unique_ptr<tflite::FlatBufferModel> _hand_model;
    std::unique_ptr<tflite::Interpreter> _hand_interpreter;
    tflite::StderrReporter _hand_error_reporter;
    // models info
    int _hand_input; // model input layer number 
    int _hand_in_height;
    int _hand_in_width;
    int _hand_in_channels;
    int _hand_in_type;

    // Input of the interpreter
    float *_pHandInputLayer;

    // outputs of the interpreter
    float *_pHandOutputLayerLandmarks;
    float *_pHandOutputLayer;
};