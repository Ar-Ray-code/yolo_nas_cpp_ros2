#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "processing.hpp"
#include "utils.hpp"

namespace yolo_nas_cpp
{


class YoloNas
{
private:
    int netInputShape[4] = {1, 3, 0, 0};
    void warmup(int round);
    // Colors colors;

public:
    cv::dnn::Net net;
    float scoreThresh;
    float iouThresh;
    std::vector<std::string> classLabels;

    PreProcessing preprocess;
    PostProcessing postprocess;
    YoloNas(std::string netPath, bool cuda, json &prepSteps, std::vector<int> imgsz, float score, float iou, std::vector<std::string> &labels);
    std::vector<Object> predict(cv::Mat &img);
};
}
