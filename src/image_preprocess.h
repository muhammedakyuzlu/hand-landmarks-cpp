#include "typedef_struct.h"
#include <opencv2/imgproc.hpp>

void normalizeZeroToOne(Pixel &pixel);
void preprocess(const cv::Mat &frame, cv::Mat &inputImg);