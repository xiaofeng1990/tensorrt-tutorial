#include "infer.hpp"
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <string>
#include <future>
#include <queue>
#include <functional>

struct Job
{
    std::shared_ptr<std::promise<std::string>> pro;
    std::string input;
};

class InferImpl : public Infer
{
public:
    virtual ~InferImpl()
    {
        stop();
    }
    void stop()
    {
        if (running_)
        {
            running_ = false;
            cv_.notify_one();
        }

        if (worker_thread_.joinable())
            worker_thread_.join();
    }
    bool startup(const std::string &file)
    {
        file_ = file;
        running_ = true; // 启动后，运行状态设置为true
        std::promise<bool> pro;
        worker_thread_ = std::thread(&InferImpl::worker, this, std::ref(pro));
        /*
            注意：这里thread 一构建好后，worker函数就开始执行了
            第一个参数是该线程要执行的worker函数，第二个参数是this指的是class InferImpl，第三个参数指的是传引用，因为我们在worker函数里要修改pro。
         */
        return pro.get_future().get();
    }
    virtual std::shared_future<std::string> commit(const std::string &input) override
    {
        Job job;
        job.input = input;
        job.pro.reset(new std::promise<std::string>);
        std::shared_future<std::string> fut = job.pro->get_future();
        {
            std::lock_guard<std::mutex> l(lock_);
            jobs_.emplace(std::move(job));
        }
        cv_.notify_one();
        return fut;
    }

    void worker(std::promise<bool> &pro)
    {
        if (file_ != "trtfile")
        {
            // failed
            pro.set_value(false);
            printf("Load model failed: %s\n", file_.c_str());
            return;
        }
        // load success
        pro.set_value(true); // 这里的promise用来负责确认infer初始化成功了
        std::vector<Job> fetched_jobs;

        while (running_)
        {
            {
                std::unique_lock<std::mutex> l(lock_);
                cv_.wait(l, [&]()
                         { return !running_ || !jobs_.empty(); }); // 一直等着，cv_.wait(lock, predicate) // 如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号

                if (!running_)
                    break; // 如果 不在运行 就直接结束循环
                int batch_size = 5;
                for (int i = 0; i < batch_size && !jobs_.empty(); ++i)
                {                                                        // jobs_不为空的时候
                    fetched_jobs.emplace_back(std::move(jobs_.front())); // 就往里面fetched_jobs里塞东西
                    jobs_.pop();                                         // fetched_jobs塞进来一个，jobs_那边就要pop掉一个。（因为move）
                }
            }

            for (auto &job : fetched_jobs)
            {
                job.pro->set_value(job.input + "---processed");
            }
            fetched_jobs.clear();
        }
        printf("Infer worker done.\n");
    }

private:
    std::atomic<bool> running_{false};
    std::string file_;
    std::thread worker_thread_;
    std::queue<Job> jobs_;
    std::mutex lock_;
    std::condition_variable cv_;
};

std::shared_ptr<Infer> create_infer(const std::string &file)
{
    /*
        [建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)]
        RAII+封装接口模式：问题定义-异常流程处理 https://v.douyin.com/NfJtnpF/
        RAII+封装接口模式：解决方案-用户友好设计 https://v.douyin.com/NfJteyc/
     */
    std::shared_ptr<InferImpl> instance(new InferImpl()); // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    if (!instance->startup(file))
    {                     // 推理器实现类实例(instance)启动。这里的file是engine file
        instance.reset(); // 如果启动不成功就reset
    }
    return instance;
}

void infer_test()
{
    auto infer = create_infer("trtfile"); // 创建及初始化 抖音网页短视频辅助讲解: 创建及初始化推理器 https://v.douyin.com/NfJvWdW/
    if (infer == nullptr)
    {
        printf("Infer is nullptr.\n");
        return;
    }
    printf("commit msg = %s\n", infer->commit("msg").get().c_str()); // 将任务提交给推理器（推理器执行commit），同时推理器（infer）也等着获取（get）结果
}