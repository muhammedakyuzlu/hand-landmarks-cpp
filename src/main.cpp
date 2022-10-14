#include "palm_detection.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "\nError! Usage: <path to tflite model> <path to input video> \n"
                  << std::endl;
        return 1;
    }

    const std::string palm_model_path = argv[1];
    const std::string video_path = argv[2];

    // const std::string image_path = "/media/test.jpg";

    PALM model;

    // conf
    model.confThreshold = 0.30;
    model.nmsThreshold = 0.40;
    model.nthreads = 4;

    // Load the saved_model
    model.loadModel(palm_model_path);

    cv::VideoCapture capture;

    // load input video or open web cam
    if (all_of(video_path.begin(), video_path.end(), ::isdigit) == false)
        capture.open(video_path); 
    else
        capture.open(stoi(video_path));

    // cv::Mat frame = cv::imread(image_path);
    // cv::Mat image;
    if (!capture.isOpened())
        throw "\nError when reading video steam\n";
    // cv::namedWindow("w", 1);

    // save video config
    bool save = false;
    // auto fourcc = capture.get(cv::CAP_PROP_FOURCC);
    // int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    // int frame_height = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    // cv::VideoWriter video(save_path, fourcc, 30, cv::Size(frame_width, frame_height), true);
    
    std::vector<cv::Rect> palm_bboxes;
    model.run(frame, palm_bboxes);
    for (;;)
    {
        capture >> frame;
        
        // image = frame.clone();;
        // frame = cv::imread("bus.jpg");
        if (frame.empty())
            break;
        // start
        // auto start = std::chrono::high_resolution_clock::now();
        // Predict on the input image
        model.run(frame, palm_bboxes);
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // std::cout << "\nModel run time 'milliseconds': " << duration.count() << "\n"
        //           << std::endl;


        // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::imshow("output", image);
        cv::waitKey(10);
    }
    capture.release();
    cv::destroyAllWindows();
}
