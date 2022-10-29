#include "image_preprocess.h"

void normalizeZeroToOne(Pixel &pixel)
{
    pixel.x = (pixel.x / 255.0);
    pixel.y = (pixel.y / 255.0);
    pixel.z = (pixel.z / 255.0);
}

void preprocess(const cv::Mat &frame, cv::Mat &inputImg)
{
    frame.convertTo(inputImg, CV_32FC3);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);
    // normalize to 0 & 1
    Pixel *pixel = inputImg.ptr<Pixel>(0, 0);
    const Pixel *endPixel = pixel + inputImg.cols * inputImg.rows;
    for (; pixel != endPixel; pixel++)
        normalizeZeroToOne(*pixel);

}