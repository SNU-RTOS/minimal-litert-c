// thread_safe_queue.hpp
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
public:
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(value);
        cv_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty() && !shutdown_) {
            cv_.wait(lock);
        }
        if (q_.empty()) return false;
        value = std::move(q_.front());
        q_.pop();
        return true;
    }

    void signal_shutdown() {
        std::lock_guard<std::mutex> lock(m_);
        shutdown_ = true;
        cv_.notify_all();
    }

private:
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool shutdown_ = false;
};
