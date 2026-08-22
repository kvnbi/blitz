#pragma once
#include <pthread.h>
#include <utility>

namespace blitz {

class NativeThread {
    static constexpr size_t StackSize = 8 * 1024 * 1024;

public:
    template <class Function, class... Args>
    explicit NativeThread(Function&& fn, Args&&... args) {
        auto* call = new std::function<void()>(
            std::bind(std::forward<Function>(fn), std::forward<Args>(args)...));

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, StackSize);
        pthread_create(&thread_, &attr, &NativeThread::entry, call);
        pthread_attr_destroy(&attr);
    }

    void join() { pthread_join(thread_, nullptr); }

private:
    static void* entry(void* arg) {
        auto* call = static_cast<std::function<void()>*>(arg);
        (*call)();
        delete call;
        return nullptr;
    }

    pthread_t thread_;
};

}
