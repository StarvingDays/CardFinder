#pragma once

#include "CardFinder.hpp"

class SubProcess : public CardFinder
{
public:

    SubProcess(int& w, int& h);

    static SubProcess& GetInstance(int& w, int& h);

    virtual auto HomomorphicCorrect(cv::Mat& src) -> cv::Mat override;

    virtual auto DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect> override;
};


SubProcess::SubProcess(int& w, int& h)
        :
        CardFinder(w, h)
{

}

SubProcess& SubProcess::GetInstance(int& w, int& h)
{
    static SubProcess instance(w, h);
    return instance;
}


//
auto SubProcess::HomomorphicCorrect(cv::Mat& src) -> cv::Mat
{
    cv::Mat log, complex;

    cv::Mat planes[] = { cv::Mat(), cv::Mat() };                                                        // 실수행렬과 허수행렬이 들어갈 planes

    src.convertTo(log, CV_32FC1);                                                                       // float 타입 변환
    cv::log((log / 255) + cv::Scalar::all(1), log);                                                     // log를 취한 영상 획득

    cv::dft(log, complex, cv::DFT_COMPLEX_OUTPUT);                                                      // 푸리에 연산 수행(2채널 실수+허수 영상 획득)

    cv::split(complex, planes);                                                                         // 영상 분할

    cv::multiply(planes[0], m_gaussian_filters[1], planes[0]);                                          // 실수부분과 가우시안 필터 곱연산
    cv::multiply(planes[1], m_gaussian_filters[1], planes[1]);                                          // 허수부분과 가우시안 필터 곱연산

    cv::merge(planes, 2, complex);                                                                      // 실수 및 허수 영상 병합

    cv::idft(complex, complex, cv::DFT_REAL_OUTPUT);                                                    // 역푸리에 연산 후 실수영상 획득

    cv::normalize(complex, complex, 0, 1, cv::NORM_MINMAX);                                             // 역푸리에 변환으로 얻는 실수영상을 0과 1로 정규화

    cv::exp(complex, complex);                                                                          // 지수함수 적용

    cv::normalize(complex, complex, 0, 255, cv::NORM_MINMAX, CV_8UC1);                                  // 0과 255로 정규화

    return complex;
}

\
auto SubProcess::DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> areas;
    int i, j, k, size = rois.size();
    int count1 = 0, count2 = 0;
    int dist = 0;

    std::sort(rois.begin(), rois.end(),                                                                 // y축을 기준으로 정렬
        [&rois](cv::Rect& l, cv::Rect& r) {
            return l.y < r.y;
        });

    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j) continue;                                                                       // i와 j가 같은 index일 경우 continue

            int gap_of_y_pos = cv::abs(rois[i].tl().y - rois[j].tl().y);

            if (gap_of_y_pos < 5 && gap_of_y_pos > -1)                                                  // roi[i]의 y축과 rois[j]의 y축의 차이가 0보다크고 5보다 작은 경우(체크카드 이름알파벳이 놓인 위치로 주정)
                areas.emplace_back(rois[j]);
        }
        if (areas.size() == 8)                                                                          // 위 조건에 해당하여 areas에 삽입된 원소 개수가 9개인 경우
        {
            areas.push_back(rois[i]);                                                                   // rois[i]를 삽입(알파벳 9자리로 추정되는 영역들을 확보)
            goto exit;                                                                                  // 2중 for문 탈출
        }
        areas.clear();                                                                                  // 위 조건을 통화할 시 areas 초기화
    }

exit:

    if (areas.size() == 9)
    {
        std::sort(areas.begin(), areas.end(),                                                           // x축 기준으로 정렬
            [&areas](cv::Rect& l, cv::Rect& r) {
                return l.x < r.x;
            });

        for (int i = 0; i < 3; ++i)
        {
            j = (i + 1) * 3;                                                                            // 9개 알파벳영역 중 3, 6번째 자리의 index

            for (k = i * 3; k < j - 1; ++k)                                                             // 0~2, 3~5, 6~8에 해당하는 k, k+1 인덱스를 순회하여 길이를 비교하는 for문
            {
                dist = GetDist(                                                                         // 알파벳간의 간격
                        cv::Point(areas[k].br().x, areas[k].tl().y), areas[k + 1].tl());

                if (dist < 5)                                                                           // 알파벳간의 간격이 5보다 작을 경우
                {
                    ++count1;                                                                           // count1을 1씩 증가
                }
            }

            if (j - 1 < 6)
            {
                dist = GetDist(                                                                         // 2-3, 5-6번째 숫자간의 간격
                        cv::Point(areas[j - 1].br().x, areas[j - 1].tl().y), areas[j].tl());            // 알파벳간 사이 간격의 조건이 참이면 count2를 1씩 증가
                if (dist > 6 && dist < 15) ++count2;

            }
            
        }

        if (count1 != 6 || count2 != 2) areas.clear();                                                  // 9자리 알파벳이 서로 여섯 쌍을 이루고 알파벳 세 개가 모인 영역 개수가 2가 아닌 경우 areas를 초기화
    }

    return areas;
}

