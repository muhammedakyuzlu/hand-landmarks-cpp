#include "hand_landmark_detection.h"
#include "typedef_struct.h"
#include <iostream>



void HandLandmark::loadModel(const std::string &hand_model_path)
{
    _hand_model = tflite::FlatBufferModel::BuildFromFile(hand_model_path.c_str(), &_hand_error_reporter);
    if (!_hand_model)
    {
        std::cout << "\nFailed to load hand model.\n"
                  << std::endl;
        exit(1);
    }
    tflite::ops::builtin::BuiltinOpResolver hand_resolver;
    tflite::InterpreterBuilder(*_hand_model.get(), hand_resolver)(&_hand_interpreter);
    TfLiteStatus status = _hand_interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to allocate the memory for tensors.\n"
                  << std::endl;
        exit(1);
    }

    // input information
    _hand_input = _hand_interpreter->inputs()[0];
    TfLiteIntArray *dims = _hand_interpreter->tensor(_hand_input)->dims;
    _hand_in_height = dims->data[1];
    _hand_in_width = dims->data[2];
    _hand_in_channels = dims->data[3];

    _hand_in_type = _hand_interpreter->tensor(_hand_input)->type;
    // input layer
    _pHandInputLayer = _hand_interpreter->typed_tensor<float>(_hand_input);

    // outputs layer
    _pHandOutputLayerLandmarks = _hand_interpreter->typed_tensor<float>(_hand_interpreter->outputs()[0]);
    _pHandOutputLayer = _hand_interpreter->typed_tensor<float>(_hand_interpreter->outputs()[1]);
    _hand_interpreter->SetNumThreads(nthreads);
}


void HandLandmark::run(cv::Mat frame,hand_landmark_result_t  &hand_result,palm_detection_result_t &palm_result){


        for (auto palm : palm_result.palms)
        {
            cv::Point topLeft;
            cv::Point bottomRight;
            topLeft.x = palm.hand_pos[0].x * _img_width;
            topLeft.y = palm.hand_pos[0].y * _img_height;
            bottomRight.x = topLeft.x + palm.hand_w * _img_width;
            bottomRight.y = topLeft.y + palm.hand_h * _img_height;
            cv::Rect _r(topLeft, bottomRight);
            cv::rectangle(frame,_r,(0,0,255),1);
        }


}










