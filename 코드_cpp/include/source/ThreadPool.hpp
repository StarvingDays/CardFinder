#pragma once

#include <memory>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <vector>


namespace ThreadPool
{
    class ThreadPool
    {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();

        // job 을 추가한다.
        template <class F, class... Args>
        std::future<typename std::result_of<F(Args...)>::type>
        PushJob(F&& f, Args&&... args);

    private:
        // 총 Worker 쓰레드의 개수.
        size_t m_thread_num;
        // Worker 쓰레드를 보관하는 벡터.
        std::vector<std::thread> m_worker;
        // 할일들을 보관하는 job 큐.
        std::queue<std::function<void()>> m_jobs;
        // 위의 job 큐를 위한 cv 와 m.
        std::condition_variable m_con_var;
        std::mutex m_job_mutex;

        // 모든 쓰레드 종료
        bool m_stop_all;

        // Worker 쓰레드
        void WorkerThread();
    };

    ThreadPool::ThreadPool(size_t num_threads)
        :
        m_thread_num(num_threads),
        m_stop_all(false)
    {
        if(m_thread_num != 0)
        {
            m_worker.reserve(m_thread_num);
            for (size_t i = 0; i < m_thread_num; ++i)
            {
                m_worker.emplace_back([this]()
                                      { this->WorkerThread(); });
            }
        }
    }

    void ThreadPool::WorkerThread()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(m_job_mutex);
            // job이 비어있지 않거나 stop_all이 true이면 lock을 걸고 wait 해제
            // 해제 시 lock() 호출

            // job이 비어있거나 stop_all이 false이면 계속 wait
            m_con_var.wait(lock, [this]()
                           { return !this->m_jobs.empty() || m_stop_all; });

            if (m_stop_all && this->m_jobs.empty())
                return;


            // 맨 앞의 job 을 뺀다.
            std::function<void()> job = std::move(m_jobs.front());
            m_jobs.pop();
            lock.unlock();

            // 해당 job 을 수행한다 :)
            job();
        }
    }

    ThreadPool::~ThreadPool()
    {
        m_stop_all = true;
        m_con_var.notify_all();

        for (auto &t : m_worker)
        {
            t.join();
        }
    }

    // class... 은 가변 길이 템플릿으로 임의의 길이의 인자들을 받을 수 있다
    // f에는 함수가들어가고 args... 부분엔 인자가 들어간다
    // result_of 특정함수의 리턴타입을 알아낸다
    // EnqueueJob 함수가 임의의 형태의 함수를 받고, 그 함수의 리턴값을 보관하는 future을 리턴한다
    // 전달받은 함수 f의 리턴값을 가지는 future를 리턴해야한다 즉, 함수 F의 리턴값은 std::result_of를 사용하면 알 수 있다
    template <class F, class... Args>
    std::future<typename std::result_of<F(Args...)>::type>
            ThreadPool::PushJob(F&& f, Args&&... args)
    {
        if (m_stop_all)
        {
            throw std::runtime_error("ThreadPool Usage Paused!");
        }

        // 임의 함수의 리턴타입의 alias(별칭)
        // 어떻게하면 void() 꼴의 함수만 저장할 수 있는 컨테이너에 특정함수를 집어 넣나?
        // void타입을 반환하는 람다함수로 래핑하는 방법이 있지만 그 방법으론 특정 리턴타입을 못 얻는다
        // 함수 f의 리턴타입을 보관하는 타입을 alias로 정의하였다
        using return_type = typename std::result_of<F(Args...)>::type;

        // 비동기적으로 실행되는 함수의 리턴값을 받아내는 packaged_task
        // 이는 함수만을 받는다
        // std::bind는 함수와 함수인자를 받아 특정시점에 호출가능하도록 하는 함수객체를 만들어준다
        // shared_ptr을 안쓰면 EnqueueJob함수가 리턴하면서 파괴가 되기때문에 예외발생함
        // [&job]() { job(); } 안에서 job 을 접근할 때 이미 그 객체는 파괴되고 없어져있다
        auto job = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            
        std::future<return_type> job_result_future = job->get_future();

        {
            std::lock_guard<std::mutex> lock(m_job_mutex);
            m_jobs.push( [job](){ (*job)(); } );
        }
        m_con_var.notify_one();

        return job_result_future;
    }

} 
