#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>


// 두 점사이의 거리를 구하는 함수
template<template<typename E> class C = cv::Point_, typename T,
    typename = typename std::enable_if<
    std::is_arithmetic<T>::value ||
    std::is_same<std::decay_t<C<T>>, cv::Point_<T>>::value>::type>
auto CalcDist(C<T> const& pt1, C<T> const& pt2)
{

    return std::sqrt(
        ((pt1.x - pt2.x) * (pt1.x - pt2.x)) +
        ((pt1.y - pt2.y) * (pt1.y - pt2.y)));
}

// 두 직선(네 개의 점)이 이루는 교점을 얻는 함수
template<template<typename E> class C = cv::Point_, typename T,
    typename = typename std::enable_if<
    std::is_arithmetic<T>::value ||
    std::is_same<std::decay_t<C<T>>, cv::Point_<T>>::value>::type>
C<T> FindCrossPoint(C<T> const& pt1, C<T> const& pt2, C<T> const& pt3, C<T> const& pt4)
{
    T xChild = {};
    T yChild = {};
    T mother = {};

    try {
        xChild = (                                                                                                      // x축 분자
            ((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.x - pt4.x)) -
            ((pt1.x - pt2.x) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));

        yChild = (                                                                                                      // y축 분자
            ((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.y - pt4.y)) -
            ((pt1.y - pt2.y) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));

        mother =
            ((pt1.x - pt2.x) * (pt3.y - pt4.y)) - ((pt1.y - pt2.y) * (pt3.x - pt4.x));                                  // 분모
    }
    catch (std::exception& ex)
    {
        return cv::Point_<T>(0, 0);
    }

    return C<T>(xChild / mother, yChild / mother);
}

// 벡터의 내적으로 각도를 구하는 함수
template<template<typename E> class C = cv::Point_, typename T,
    typename = typename std::enable_if<
    std::is_arithmetic<T>::value ||
    std::is_same<std::decay_t<C<T>>, cv::Point_<T>>::value>::type>
T FindAngle(C<T> const& pt1, C<T> const& pt2)
{
    T len1 = std::sqrt(std::pow(pt1.x, 2) + std::pow(pt1.y, 2));                                                        // 두 점사이의 거리를 계산
    T len2 = std::sqrt(std::pow(pt2.x, 2) + std::pow(pt2.y, 2));
    T dotProduct = (pt1.x * pt2.x) + (pt1.y * pt2.y);                                                                   // 벡터 내적 획득
    T crossProduct = (pt1.x * pt2.y) - (pt1.y * pt2.x);                                                                 // 벡터 외적 획득

    T cosine = dotProduct / (len1 * len2);                                                                              // 코사인 각 획득
    T angle_degree = std::acos(cosine) * (180.0f / CV_PI);                                                              // 아크코사인으로 각도 획득

    if (crossProduct < 0) angle_degree = -angle_degree;                                                                 // 벡터 외적이 0보다 작으면 -값으로 변환


    return angle_degree;
}


// 90도로 영상을 회전시키는 함수
template<typename T, int CV_TYPE, typename = typename
        std::enable_if_t<std::is_same<T, uchar>::value ||
        std::is_same<T, cv::Vec3b>::value>>
cv::Mat Rotate90(cv::Mat& src)
{
    static_assert(
            (std::is_same<T, uchar>::value && std::integral_constant<int, CV_TYPE>::value == CV_8UC1) ||                // 타입이 uchar and CV_8UC1 이거나
            (std::is_same<T, cv::Vec3b>::value && std::integral_constant<int, CV_TYPE>::value == CV_8UC3),              // 타입이 Vec3b and CV_8UC3가 아니면 컴파일 에러
            "typename T and cv_type must be uchar-CV_8UC1 or Vec3b-CV_8UC3");

    cv::Mat dst = cv::Mat::zeros(cv::Size(src.rows, src.cols), CV_TYPE);

    int& w = src.cols;
    int& h = src.rows;

    int x, y;

    for (x = 0; x < w; ++x)
    {
        T* ptr = dst.ptr<T>(x);
        for (y = 0; y < h; y++)
        {
            ptr[y] = src.ptr<T>(h - 1 - y)[x];                                                                          // 각 픽셀들을 90도로 회전된 자리에 삽입
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

    cv::Mat q0(src, cv::Rect(0, 0, midCol + isColOdd, midRow + isRowOdd));                                             // 1사분면 영상
    cv::Mat q1(src, cv::Rect(midCol + isColOdd, 0, midCol, midRow + isRowOdd));                                        // 2사분면 영상
    cv::Mat q2(src, cv::Rect(0, midRow + isRowOdd, midCol + isColOdd, midRow));                                        // 3사분면 영상
    cv::Mat q3(src, cv::Rect(midCol + isColOdd, midRow + isRowOdd, midCol, midRow));                                   // 4사분면 영상

    if (!(isColOdd || isRowOdd))                                                                                       // cols와 rows 둘 중하나의 길이가 짝수인 경우
    {
        cv::Mat tmp;                                                                                                   // 1사분면과 3사분면을 교환
        q0.copyTo(tmp);
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                                                                                                // 2사분면과 4사분면을 교환
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
    int big = std::max(src.cols, src.rows);                                                                             // 가로 세로 길이 중 큰 값으로 반환

    cv::Mat result(big * 1.5f, big * 1.5f, src.type(), cv::Scalar(0));                                                  // 정사각형 영상 생성

    cv::Point start = (result.size() - src.size()) / 2;                                                                 // 정사각형 영상에서 숫자 이미지를 위치시킬 시작지점을

    src.copyTo(                                                                                                         // 숫자이미지를 정사각형 0행렬 영상에 붙여넣기
        result(cv::Rect(start, src.size())));

    cv::resize(result, result, cv::Size(20, 20), cv::INTER_LANCZOS4);                                                   // 28 X 28 사이즈로 변경

    return result;
}

// 스레드의 개수를 반환하는 함수
int GetThreadInfo()
{
    int threadNum = cv::getNumThreads();                                                                                // 스레드 개수 획득


    if (threadNum > 16)
        threadNum = 8;
    else if (threadNum == 8)
        threadNum = 4;
    else if (threadNum == 16)
        threadNum = 8;

    return threadNum;
}
