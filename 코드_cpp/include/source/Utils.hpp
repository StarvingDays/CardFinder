#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <tuple>

// 두 점사이의 거리를 구하는 함수
template<typename T>
T GetDist(cv::Point_<T> pt1, cv::Point_<T> pt2)
{
    return std::sqrt(
            ((pt1.x - pt2.x) * (pt1.x - pt2.x)) +
            ((pt1.y - pt2.y) * (pt1.y - pt2.y)));
}

// 벡터의 내적으로 각도를 구하는 함수
template<typename T>
T GetAngleFromDotProduct(const cv::Point_<T> pt1, const cv::Point_<T> pt2)
{
    T len1 = std::sqrt(std::pow(pt1.x, 2) + std::pow(pt1.y, 2));
    T len2 = std::sqrt(std::pow(pt2.x, 2) + std::pow(pt2.y, 2));
    T dotProduct = (pt1.x * pt2.x) + (pt1.y * pt2.y);
    T cosine = dotProduct / (len1 * len2);

    return std::acos(cosine) * (180 / CV_PI);
}

// 두 점으로 각도를 구하는 함수
template<typename T>
T GetAngleFromTwoPoint(const cv::Point_<T> pt1, const cv::Point_<T> pt2)
{
    double dy = pt2.y - pt1.y;
    double dx = pt2.x - pt1.x;
    double angle = atan(dy / dx) * (180.0 / CV_PI);

    return angle;
}

// 90도로 영상을 회전시키는 함수
 cv::Mat Rotate90(cv::Mat& src)
 {
     if(src.empty()) return cv::Mat();

     cv::Mat dst = cv::Mat::zeros(cv::Size(src.rows, src.cols), CV_8UC1);

     int& w = src.cols;
     int& h = src.rows;

     int i, j;

     for(j = 0; j < w; ++j)
     {
         uchar* ptr = dst.ptr<uchar>(j);
         for(i = 0; i < h; i++)
         {
             ptr[i] = src.ptr<uchar>(h - 1 - i)[j];
         }
     }

     return dst;
 }

 // 이미지의 1,2,3,4분 면을 취하여 변환하는 함수 
void Shift(cv::Mat& src)
{
    int midCol = src.cols >> 1;
    int midRow = src.rows >> 1;

    int isColOdd = src.cols % 2 == 1;
    int isRowOdd = src.rows % 2 == 1;

    cv::Mat q0(src, cv::Rect(0, 0, midCol + isColOdd, midRow + isRowOdd));
    cv::Mat q1(src, cv::Rect(midCol + isColOdd, 0, midCol, midRow + isRowOdd));
    cv::Mat q2(src, cv::Rect(0, midRow + isRowOdd, midCol + isColOdd, midRow));
    cv::Mat q3(src, cv::Rect(midCol + isColOdd, midRow + isRowOdd, midCol, midRow));

    if (!(isColOdd || isRowOdd))
    {
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }
    else
    {
        cv::Mat tmp0, tmp1, tmp2, tmp3;
        q0.copyTo(tmp0);
        q1.copyTo(tmp1);
        q2.copyTo(tmp2);
        q3.copyTo(tmp3);

        tmp0.copyTo(src(cv::Rect(midCol, midRow, midCol + isColOdd, midRow + isRowOdd)));
        tmp3.copyTo(src(cv::Rect(0, 0, midCol, midRow)));

        tmp1.copyTo(src(cv::Rect(0, midRow, midCol, midRow + isRowOdd)));
        tmp2.copyTo(src(cv::Rect(midCol, 0, midCol + isColOdd, midRow)));
    }
}

// 숫자 객체를 정사각형 이미지의 중앙으로 오게 만드는 함수
cv::Mat PlaceMiddle(cv::Mat src)
{
    int big = std::max(src.cols, src.rows);

    cv::Mat result(big * 1.5f, big * 1.5f, src.type(), cv::Scalar(0));

    cv::Point start = (result.size() - src.size()) / 2;

    src.copyTo(
        result(cv::Rect(start, src.size())));

    cv::resize(result, result, cv::Size(20, 20), cv::INTER_LANCZOS4);

    return result;
}

// 스레드의 개수를 반환하는 함수
int GetThreadInfo()
{
    int threadNum = cv::getNumThreads();

    if (threadNum > 16)
        threadNum = 8;
    else if (threadNum == 8)
        threadNum = 4;
    else if (threadNum == 16)
        threadNum = 8;

    return threadNum;
}
