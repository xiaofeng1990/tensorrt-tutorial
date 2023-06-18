#include <thread>
#include <vector>
#include <future>
#include <chrono>

using namespace std;
void future_test()
{
    promise<int> pro;
    shared_future<int> fut = pro.get_future();
    std::thread(
        [&]()
        {
            printf("Async thread start.\n");
            this_thread::sleep_for(chrono::seconds(5));
            printf("Set value to 555.\n");
            pro.set_value(555);
            printf("Set value done.\n");
        })
        .detach();
    printf("Wait value.\n");
    printf("fut.get() = %d\n", fut.get());
}