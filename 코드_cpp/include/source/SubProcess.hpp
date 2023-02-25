#pragma once

#include "CardFinder.hpp"

class SubProcess : public CardFinder
{
public:

    SubProcess(int& h, int& w);

    static SubProcess& GetInstance(int h, int w);

    virtual auto HomomorphicCorrect(cv::Mat& src) -> cv::Mat override;

    virtual auto AreaSegmant(cv::Mat& src) -> std::vector<cv::Rect> override;
};


SubProcess::SubProcess(int& h, int& w)
        :
        CardFinder(h, w)
{

}

SubProcess& SubProcess::GetInstance(int h, int w)
{
    static SubProcess instance(h, w);
    return instance;
}



auto SubProcess::HomomorphicCorrect(cv::Mat& src) -> cv::Mat
{
    cv::Mat log, complex;

    cv::Mat planes[] = { cv::Mat(), cv::Mat() };

    src.convertTo(log, CV_32FC1);
    cv::log((log / 255) + cv::Scalar::all(1), log);

    cv::dft(log, complex, cv::DFT_COMPLEX_OUTPUT);

    cv::split(complex, planes);

    cv::multiply(planes[0], m_gaussian_filters[1], planes[0]);
    cv::multiply(planes[1], m_gaussian_filters[1], planes[1]);


    cv::merge(planes, 2, complex);

    cv::idft(complex, complex, cv::DFT_REAL_OUTPUT);

    return complex;
}




auto SubProcess::AreaSegmant(cv::Mat& src) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> loc;

    try
    {
        cv::Mat labels, stats, centroid;

        int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroid, 8, CV_32SC1);

        int* mStats_ptr = nullptr;
        for (int j = 0; j < numLabels; ++j)
        {

            mStats_ptr = stats.ptr<int>(j);

            int& area = mStats_ptr[cv::CC_STAT_LEFT];
            int& left = mStats_ptr[cv::CC_STAT_LEFT];
            int& top = mStats_ptr[cv::CC_STAT_TOP];
            int& width = mStats_ptr[cv::CC_STAT_WIDTH];
            int& height = mStats_ptr[cv::CC_STAT_HEIGHT];

            cv::Rect rect(cv::Point(left, top), cv::Point(left + width, top + height));

            if ((height > 6 && height < 100) && (height > width))
                loc.emplace_back(rect);

        }

    }
    catch(...)
    {

    }

    return loc;
}

