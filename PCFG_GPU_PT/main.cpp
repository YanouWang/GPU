#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下
// nvcc main.cpp train.cpp guessing.cu md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./output/results.txt");
    const int pt_batch_size = 8; // 一次批量处理8个PT，可根据显存调整
while (!q.priority.empty())
{
    // 1. 批量取出pt_batch_size个PT
    std::vector<PT> pts_this_batch;
    for (int i = 0; i < pt_batch_size && !q.priority.empty(); ++i) {
        pts_this_batch.push_back(q.priority.front());
        q.priority.erase(q.priority.begin());
    }

    // 2. GPU批量生成所有PT的猜测
    std::vector<std::string> batch_guesses;
    q.BatchGenerate(pts_this_batch, batch_guesses);

    // 3. 合并到主guesses
    q.guesses.insert(q.guesses.end(), batch_guesses.begin(), batch_guesses.end());
    q.total_guesses = q.guesses.size();

    if (q.total_guesses - curr_num >= 100000)
    {
        cout << "Guesses generated: " << history + q.total_guesses << endl;
        curr_num = q.total_guesses;

        int generate_n = 10000000;
        if (history + q.total_guesses > generate_n)
        {
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
            cout << "Hash time:" << time_hash << "seconds" << endl;
            cout << "Train time:" << time_train << "seconds" << endl;
            break;
        }
    }

    if (curr_num > 1000000)
    {
        auto start_hash = system_clock::now();
        bit32 state[4];
        for (string pw : q.guesses)
        {
            MD5Hash(pw, state);
        }
        auto end_hash = system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

        history += curr_num;
        curr_num = 0;
        q.guesses.clear();
    }
}
}