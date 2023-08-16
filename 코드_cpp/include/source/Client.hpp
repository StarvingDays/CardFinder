#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/detail/base64.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace http = boost::beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;
namespace asio = boost::asio;


class Client
{
public:
    Client() = delete;

    Client(const char* host, const char* port, const char* target, int http_version);

    ~Client() = default;

    auto SetData(cv::Mat& src) -> void;                                                     // 이미지 데이터를 base64 포멧의 string타입으로 변환하는 함수
    auto IsBufferEmpty() -> bool;                                                           // base64 타입으로 변환된 버퍼가 비어있는 여부를 확인하는 함수
    auto ResetBuffer() -> void;                                                             // base64 타입으로 변환된 버퍼를 비우는 함수
    auto StartAnalysis() -> void;                                                           // 웹서버 통신으로 카드 이미지를 분석하는 함수
    auto GetMessage() -> std::string;                                                       // 웹서버에서 읽어온 reponse 패킷의 body(이미지 분석 결과)를 저장하는 함수
    auto Connect() -> bool;                                                                 // 서버 연결 함수



private:
    auto Run() -> void;
    auto Read() -> void;                                                                    // 웹 서버로 부터 response을 받아 body 데이터를 읽는 함수
    auto Write() -> void;                                                                   // 웹 서버로 request 패킷을 전송하는 함수


    asio::io_context m_io_context;

    tcp::socket m_socket{ m_io_context };

    std::string m_host, m_port, m_target, m_url_host;

    int m_http_version;

    std::chrono::steady_clock::duration m_timeout;

    std::string m_img_data_buffer, m_res_mes;
};

Client::Client(const char* host, const char* port, const char* target, int http_version)
        :
        m_host(host),                                                                       // ip 주소
        m_port(port),                                                                       // 포트 번호
        m_target(target),                                                                   // url 타겟
        m_url_host(host + std::string(":") + port),                                         // ip + port + target
        m_http_version(http_version),                                                       // http 버전
        m_timeout(std::chrono::seconds(10))                                                 // 타임아웃 타이머
{

}


bool Client::Connect()
{
    auto endpoints = tcp::resolver(m_io_context).resolve(m_host, m_port);                   // 엔드포인트 생성 0.0.0.0:9999
    bool is_connected;                                                                      // 서버 연결 확인 bool 값

    asio::async_connect(m_socket, endpoints,                                                // 서버 연결 시도
        [&](std::error_code res_error, asio::ip::tcp::endpoint endpoint)
        {
            if (res_error)
            {
                is_connected = false;                                                       // 서버연결 실패
                m_res_mes = "Server is closed!";
            }
            else
                is_connected = true;                                                        // 서버연결 성공

        });


    Run();                                                                                  // 서버 연결 시 데드라인 타이머를 설정하여 특정 시간이 지나도 서버 연결이 지연 될 시 연결 시도를 종료한다

    return is_connected;
}


auto Client::Run() -> void
{
    m_io_context.restart();

    m_io_context.run_for(m_timeout);                                                        // io_context 타이머 대기 :

    if (!m_io_context.stopped())                                                            // io_context stop
    {
        m_socket.close();

        m_io_context.run();
    }

}

auto Client::Write() -> void
{
    if (!m_img_data_buffer.empty())
    {
        http::request<http::string_body> req                                                // http 리퀘스트 패킷 작성전용 자료구조
        {
            boost::beast::http::verb::post, m_target, m_http_version                        // POST /OCR HTTP/1.1
        };
        req.set(http::field::host, m_url_host);                                             // Host: 192.168.0.21:2005
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);                       // User-Agent: Boost.Beast/347
        req.set(http::field::content_type, "application/json");                             // Content-Type: application/json
        req.body() = std::move(m_img_data_buffer);                                          // ImageData
        req.prepare_payload();                                                              // Content-Length: ${}

        boost::beast::http::async_write(m_socket, req,                                      // request 패킷을 서버로 전달
            [&, this](std::error_code& res_error, std::size_t bytes_transferred)
            {
                if (res_error) m_res_mes = "Write Failed!";
            });

        Run();                                                                              // 송신완료 될 때까지 대기
    }
}


auto Client::Read() -> void
{
    boost::beast::flat_buffer buffer;                                                       // 일기 버퍼
    http::response<http::dynamic_body> res;                                                 // response 동적 패킷

    http::async_read(m_socket, buffer, res,                                                 // 서버로 부터 버퍼를 읽어온다
         [&](std::error_code& res_error, std::size_t bytes_transferred)
         {
             if (!res_error)
                 m_res_mes = boost::beast::buffers_to_string(res.body().data());            // 읽기 완료 시 response 패킷의 body를 저장
             else
                 m_res_mes = res_error.message();
         });

    Run();                                                                                  // 읽기 완료 될 때까지 대기
}

auto Client::SetData(cv::Mat& src) -> void
{
    std::vector<uchar> buf;

    cv::imencode(".png", src, buf);                                                         // 2차원 배열 Mat 데이터를 utf-8 형식의 1차원 버퍼로 변환한다

    m_img_data_buffer.resize(                                                               // base64 버퍼 : buf 크기 만큼 저장공간 생성
            boost::beast::detail::base64::encoded_size(buf.size()));

    std::size_t encodedSize = boost::beast::detail::base64::encode(                         // m_img_data_buffer의 시작지점에서 buf의 처음부터 끝지점까지의 데이터를 base64 포멧으로 변환하여 저장
            const_cast<char*>(m_img_data_buffer.data()),
            buf.data(),
            buf.size()
    );

    m_img_data_buffer.resize(encodedSize);                                                  // 실제 인코딩 크기에 맞추기 위해 인코딩 스트링 데이터의 크기를 변환

    m_img_data_buffer = "{\"Checkcard\":\"" + m_img_data_buffer + "\"}";                    // json 포멧으로 변환

}

auto Client::IsBufferEmpty() -> bool
{
    return m_img_data_buffer.empty();
}

auto Client::ResetBuffer() -> void
{
    m_img_data_buffer.clear();
}

auto Client::StartAnalysis() -> void
{
    Write();
    Read();
}

auto Client::GetMessage() -> std::string
{
    return std::move(m_res_mes);
}
