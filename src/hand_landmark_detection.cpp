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
    _pHandOutputLayerScore = _hand_interpreter->typed_tensor<float>(_hand_interpreter->outputs()[1]);
    _hand_interpreter->SetNumThreads(nthreads);
}


void HandLandmark::run(cv::Mat frame,std::vector<hand_landmark_result_t>  &hand_results,palm_detection_result_t &palm_result){
    



    for (auto palm : palm_result.palms )
    {        

        
        float small_x = _img_width;
        float small_y = _img_height;
        float large_x = 0;
        float large_y = 0;

        for(int ll =0 ; ll<4;ll++){
            small_x = palm.hand_pos[ll].x < small_x ? palm.hand_pos[ll].x : small_x;
            small_y = palm.hand_pos[ll].y < small_y ? palm.hand_pos[ll].y : small_y;
            large_x = palm.hand_pos[ll].x > large_x ? palm.hand_pos[ll].x : large_x;
            large_y = palm.hand_pos[ll].y > large_y ? palm.hand_pos[ll].y : large_y;
        }

        int x = small_x * _img_width;
        int y = small_y * _img_height ;
        int w = (large_x - small_x)  * _img_width;
        int h = (large_y - small_y)  * _img_height;

        x = (int) x > 0 ? x : 0 ;
        y = (int) y > 0 ? y : 0 ;
        w = (int) w + x < _img_width  ? w : _img_width   - x;
        h = (int) h + y < _img_height ? h : _img_height  - y;

        cv::Mat croppedImage ;
        frame(cv::Rect(x,y,w,h)).copyTo(croppedImage);
        if (croppedImage.cols == 0 || croppedImage.rows == 0 )
            return;

        cv::Mat handInputImg;
        cv::resize(croppedImage, handInputImg, cv::Size(_hand_in_width, _hand_in_height));

        // flatten rgb image to input layer.
        memcpy(_pHandInputLayer, handInputImg.ptr<float>(0),
           _hand_in_width * _hand_in_height * _hand_in_channels * sizeof(float));

        // Inference
        TfLiteStatus status = _hand_interpreter->Invoke();
        if (status != kTfLiteOk)
        {
            std::cout << "\nFailed to run inference!!\n";
            exit(1);
        }
        hand_landmark_result_t _tem = {};
        
        _tem.score = _pHandOutputLayerScore[0];

        if(_pHandOutputLayerScore[0] > confThreshold){
            std::cout << "score  " <<_pHandOutputLayerScore[0] << std::endl;
            for (int i = 0; i < HAND_JOINT_NUM; i ++)
            {
                _tem.joint[i].x = _pHandOutputLayerLandmarks[3 * i + 0] / (float)_hand_in_width;
                _tem.joint[i].y = _pHandOutputLayerLandmarks[3 * i + 1] / (float)_hand_in_height;
                _tem.joint[i].z = _pHandOutputLayerLandmarks[3 * i + 2] / (float)_hand_in_width;

                _tem.joint[i].x *= w ;
                _tem.joint[i].y *= h ;
                _tem.joint[i].z *= w ;
            }
            _tem.offset.x = x;
            _tem.offset.y = y;
            hand_results.push_back(_tem);
        }
    }

}






// cv::Point topLeft;
// cv::Point bottomRight;
// topLeft.x = palm.hand_pos[0].x * _img_width;
// topLeft.y = palm.hand_pos[0].y * _img_height;
// bottomRight.x = topLeft.x + palm.hand_w * _img_width;
// bottomRight.y = topLeft.y + palm.hand_h * _img_height;
// cv::Rect _r(topLeft, bottomRight);
// cv::rectangle(frame,_r,(0,0,255),1);









