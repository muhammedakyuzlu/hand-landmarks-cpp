#include "palm_detection.h"

void PALM::loadModel(const  std::string path)
{

    _model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!_model)
    {
        std::cout << "\nFailed to load the model.\n" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nModel loaded successfully.\n";
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);
    TfLiteStatus status = _interpreter->AllocateTensors();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to allocate the memory for tensors.\n"
                  << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "\nMemory allocated for tensors.\n";
    }

    // input information
    _input = _interpreter->inputs()[0];
    TfLiteIntArray *dims = _interpreter->tensor(_input)->dims;
    _in_height = dims->data[1];
    _in_width = dims->data[2];
    _in_channels = dims->data[3];
    _in_type = _interpreter->tensor(_input)->type;
    _input_ = _interpreter->typed_tensor<float_t>(_input);
    _interpreter->SetNumThreads(nthreads);
}



void PALM::preprocess(cv::Mat &image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(_in_height, _in_width), cv::INTER_CUBIC);
    // image.convertTo(image, CV_8U); // the image should be float 32
    image.convertTo(image,  CV_32F);    //, 1.0/255`
    
    // image -= 127.5;
    // image /= 127.5;



    // 0 255  
    // (X - 127.5 ) / 127.5
    // X = 0 ---> -1
    // X = 255 ---> 1


}

template <typename T>
void PALM::fill(T *in, cv::Mat &src)
{
    int n = 0, nc = src.channels(), ne = src.elemSize();
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            for (int c = 0; c < nc; ++c)
                in[n++] = src.data[y * src.step + x * ne + c];
}

std::vector<std::vector<float>> PALM::tensorToVector2D(TfLiteTensor *pOutputTensor, const int &row, const int &colum)
{
    std::vector<std::vector<float>> v;
    for (int i = 0; i < row; i++)
    {
        std::vector<float> _tem;
        for (int j = 0; j < colum; j++)
        {
            float val_float = (float)pOutputTensor->data.f[i * colum + j];
            _tem.push_back(val_float);
        }
        v.push_back(_tem);
    }
    return v;
}

std::vector<float> PALM::tensorToVector1D(TfLiteTensor *pOutputTensor, const int &row)
{
    std::vector<float> v;
    for (int j = 0; j < row; j++)
    {
        float val_float = (float)pOutputTensor->data.f[row + j];
        v.push_back(val_float);
    }

    return v;
}


void PALM::nonMaximumSuppression(std::vector<std::vector<float>> pred_bbox, std::vector<float> pred_anchor, std::vector<cv::Rect> &bboxes){


    
    for (int i = 0; i < pred_bbox.size(); i++)
    {

    for (int j = 0; j < pred_bbox[0].size(); j++)
    {
        std::cout << pred_bbox[i][j] << " ";
    }
    std::cout<< std::endl;
    }


    // for (int j = 0; j < pred_anchor.size(); j++)
    // {
    //     std::cout << pred_anchor[j] << " ";
    // }
    //     for (int i = 0; i < pred_bbox.size(); i++)
    // {

    // for (int j = 0; j < pred_bbox[0].size(); j++)
    // {
    //     std::cout << pred_bbox[i][j] << " ";
    // }
    // std::cout<< std::endl;
    // }

    std::cout<< std::endl;
    std::cout<< std::endl;
    std::cout<< std::endl;
    std::cout<< std::endl;
    

// out_reg shape is [number of anchors, 18]
// Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
// Second dimension 4 - 18 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7

}

void PALM::run(cv::Mat frame, std::vector<cv::Rect> &bboxes)
{
    
    _img_height = frame.rows;
    _img_width = frame.cols;
    
    preprocess(frame);
    fill(_input_, frame);
    

    // Inference
    TfLiteStatus status = _interpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "\nFailed to run inference!!\n";
        exit(1);
    }

    int _out_bbox   = _interpreter->outputs()[0];
    TfLiteIntArray *_out_dims_bbox   = _interpreter->tensor(_out_bbox)->dims;
    // --> 1 2016 18
    int _out_anchor = _interpreter->outputs()[1];
    TfLiteIntArray *_out_dims_anchor = _interpreter->tensor(_out_anchor)->dims;
    // --> 1 2016 0
    // std::cout <<_out_dims_bbox->data[0]<<" "<<_out_dims_bbox->data[1] <<" "<<_out_dims_bbox->data[2] << std::endl;
    // std::cout <<_out_dims_anchor->data[0]<<" "<<_out_dims_anchor->data[1] <<" "<<_out_dims_anchor->data[2] << std::endl;



    int _out_row_bbox   = _out_dims_bbox->data[1];   
    int _out_colum_bbox = _out_dims_bbox->data[2]; 
    TfLiteTensor *pOutputTensor_bbox = _interpreter->tensor(_interpreter->outputs()[0]);
    std::vector<std::vector<float>> pred_bbox = tensorToVector2D(pOutputTensor_bbox, _out_row_bbox, _out_colum_bbox);

    int _out_row_anchor = _out_dims_anchor->data[1];
    TfLiteTensor *pOutputTensor_anchor = _interpreter->tensor(_interpreter->outputs()[1]);
    std::vector<float> pred_anchor = tensorToVector1D(pOutputTensor_anchor, _out_row_anchor);
    

    nonMaximumSuppression(pred_bbox,pred_anchor,bboxes);

};