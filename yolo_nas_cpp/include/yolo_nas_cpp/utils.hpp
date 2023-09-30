#pragma once

#include <vector>
#include <fstream>
#include "color.hpp"
#include "coco_labels.hpp"
#include <opencv2/opencv.hpp>

namespace yolo_nas_cpp
{

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


std::string LogInfo(std::string header, std::string body);

std::string LogWarning(std::string header, std::string body);

std::string LogError(std::string header, std::string body);

void exists(std::string path);

bool isNumber(const std::string &s);

std::vector<std::string> read_class_labels_file(std::string file_name);
void draw_objects(cv::Mat bgr, const std::vector<Object>& objects, const std::vector<std::string>& class_names);

class VideoExporter
{
private:
    cv::VideoWriter writer;

public:
    std::string exportPath;

    VideoExporter(cv::VideoCapture &cap, std::string path);
    void write(cv::Mat &frame);
    void close();
};
} // namespace yolo_nas_cpp
