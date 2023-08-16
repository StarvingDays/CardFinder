#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cmath>
#include <mutex>



// �� �������� �Ÿ��� ���ϴ� �Լ�
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

// �� ����(�� ���� ��)�� �̷�� ������ ��� �Լ�
template<template<typename E> class C = cv::Point_, typename T,
        typename = typename std::enable_if<
                std::is_arithmetic<T>::value ||
                std::is_same<std::decay_t<C<T>>, cv::Point_<T>>::value>::type>
C<T> FindCrossPoint(C<T> const& pt1, C<T> const& pt2, C<T> const& pt3, C<T> const& pt4)
{
    T xChild = {};
    T yChild = {};
    T mother = {};

    try
    {
        xChild = (((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.x - pt4.x)) -                                              // x�� ����
                 ((pt1.x - pt2.x) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));

        yChild = (((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.y - pt4.y)) -                                              // y�� ����
                 ((pt1.y - pt2.y) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));

        mother = ((pt1.x - pt2.x) * (pt3.y - pt4.y)) - ((pt1.y - pt2.y) * (pt3.x - pt4.x));                             // �и�
    }
    catch (std::exception& ex)
    {
        return cv::Point_<T>(0, 0);
    }

    return C<T>(xChild / mother, yChild / mother);
}

// ������ �������� ������ ���ϴ� �Լ�
template<template<typename E> class C = cv::Point_, typename T,
        typename = typename std::enable_if<
                std::is_arithmetic<T>::value ||
                std::is_same<std::decay_t<C<T>>, cv::Point_<T>>::value>::type>
T FindAngle(C<T> const& pt1, C<T> const& pt2)
{
    T len1 = std::sqrt(std::pow(pt1.x, 2) + std::pow(pt1.y, 2));                                                        // �� �������� �Ÿ��� ���
    T len2 = std::sqrt(std::pow(pt2.x, 2) + std::pow(pt2.y, 2));
    T dotProduct = (pt1.x * pt2.x) + (pt1.y * pt2.y);                                                                   // ���� ���� ȹ��
    T crossProduct = (pt1.x * pt2.y) - (pt1.y * pt2.x);                                                                 // ���� ���� ȹ��

    T cosine = dotProduct / (len1 * len2);                                                                              // �ڻ��� �� ȹ��
    T angle_degree = std::acos(cosine) * (180.0f / CV_PI);                                                              // ��ũ�ڻ������� ���� ȹ��

    if (crossProduct < 0) angle_degree = -angle_degree;                                                                 // ���� ������ 0���� ������ -������ ��ȯ


    return angle_degree;
}


// 90���� ������ ȸ����Ű�� �Լ�
template<typename T, int CV_TYPE, typename = typename
std::enable_if_t<std::is_same<T, uchar>::value ||
                 std::is_same<T, cv::Vec3b>::value>>
cv::Mat Rotate90(cv::Mat& src)
{
    static_assert(
            (std::is_same<T, uchar>::value && std::integral_constant<int, CV_TYPE>::value == CV_8UC1) ||                // Ÿ���� uchar and CV_8UC1 �̰ų�
            (std::is_same<T, cv::Vec3b>::value && std::integral_constant<int, CV_TYPE>::value == CV_8UC3),              // Ÿ���� Vec3b and CV_8UC3�� �ƴϸ� ������ ����
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
            ptr[y] = src.ptr<T>(h - 1 - y)[x];                                                                          // �� �ȼ����� 90���� ȸ���� �ڸ��� ����
        }
    }

    return dst;
}

// �̹����� 1,2,3,4�� ���� ���Ͽ� ��ȯ�ϴ� �Լ�
void Shift(cv::Mat& src)
{
    int midCol = src.cols >> 1;
    int midRow = src.rows >> 1;

    int isColOdd = src.cols % 2 == 1;
    int isRowOdd = src.rows % 2 == 1;

    cv::Mat q0(src, cv::Rect(0, 0, midCol + isColOdd, midRow + isRowOdd));                                             // 1��и� ����
    cv::Mat q1(src, cv::Rect(midCol + isColOdd, 0, midCol, midRow + isRowOdd));                                        // 2��и� ����
    cv::Mat q2(src, cv::Rect(0, midRow + isRowOdd, midCol + isColOdd, midRow));                                        // 3��и� ����
    cv::Mat q3(src, cv::Rect(midCol + isColOdd, midRow + isRowOdd, midCol, midRow));                                   // 4��и� ����

    if (!(isColOdd || isRowOdd))                                                                                       // cols�� rows �� ���ϳ��� ���̰� ¦���� ���
    {
        cv::Mat tmp;                                                                                                   // 1��и�� 3��и��� ��ȯ
        q0.copyTo(tmp);
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                                                                                                // 2��и�� 4��и��� ��ȯ
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


