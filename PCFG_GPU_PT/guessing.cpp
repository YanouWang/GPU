#include "PCFG.h"
#include <cuda_runtime.h>
#include <sstream>
using namespace std;

__global__ void copy_strings(int N, const char* flat_values, const int* offsets, const int* lengths, const int* out_offsets, char* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int start = offsets[i];
        int len = lengths[i];
        int out_start = out_offsets[i];
        for (int j = 0; j < len; ++j) {
            output[out_start + j] = flat_values[start + j];
        }
    }
}

__global__ void concat_guess(
    int N, const char* guess, int guess_len,
    const char* flat_values, const int* offsets, const int* lengths, const int* out_offsets, char* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int start = offsets[i];
        int len = lengths[i];
        int out_start = out_offsets[i];
        // 先拷贝guess
        for (int j = 0; j < guess_len; ++j)
            output[out_start + j] = guess[j];
        // 再拷贝value
        for (int j = 0; j < len; ++j)
            output[out_start + guess_len + j] = flat_values[start + j];
    }
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        int N = a->ordered_values.size();
        const int batch = 100000; // 每批最多处理10万条，防止资源耗尽
        for (int base = 0; base < N; base += batch) {
            int curN = std::min(batch, N - base);

            // 分批准备 offsets, lengths, out_offsets, flat_values
            std::vector<int> offsets(curN), lengths(curN), out_offsets(curN);
            std::vector<char> flat_values;
            int offset = 0, total_output_len = 0;
            for (int i = 0; i < curN; ++i) {
                const auto& s = a->ordered_values[base + i];
                offsets[i] = offset;
                lengths[i] = s.size();
                flat_values.insert(flat_values.end(), s.begin(), s.end());
                offset += s.size();
            }
            for (int i = 0; i < curN; ++i) {
                out_offsets[i] = total_output_len;
                total_output_len += lengths[i];
            }

            // 分配并拷贝到GPU
            int *d_offsets, *d_lengths, *d_out_offsets;
            char *d_flat_values, *d_output;
            cudaMalloc(&d_offsets, curN * sizeof(int));
            cudaMalloc(&d_lengths, curN * sizeof(int));
            cudaMalloc(&d_out_offsets, curN * sizeof(int));
            cudaMalloc(&d_flat_values, flat_values.size() * sizeof(char));
            cudaMalloc(&d_output, total_output_len * sizeof(char));
            cudaMemcpy(d_offsets, offsets.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lengths, lengths.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_offsets, out_offsets.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_flat_values, flat_values.data(), flat_values.size() * sizeof(char), cudaMemcpyHostToDevice);

            // 启动kernel
            int threads = 256;
            int blocks = (curN + threads - 1) / threads;
            copy_strings<<<blocks, threads>>>(curN, d_flat_values, d_offsets, d_lengths, d_out_offsets, d_output);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
                exit(1);
            }
            cudaDeviceSynchronize();

            // 拷贝回CPU
            std::vector<char> output(total_output_len);
            cudaMemcpy(output.data(), d_output, total_output_len * sizeof(char), cudaMemcpyDeviceToHost);

            // 还原字符串
            for (int i = 0; i < curN; ++i) {
                guesses.emplace_back(output.data() + out_offsets[i], lengths[i]);
            }
            total_guesses += curN;

            // 释放GPU内存
            cudaFree(d_offsets);
            cudaFree(d_lengths);
            cudaFree(d_flat_values);
            cudaFree(d_output);
            cudaFree(d_out_offsets);
        }
        /*
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
        */
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的

        int N = a->ordered_values.size();
        std::vector<char> guess_chars(guess.begin(), guess.end());
        int guess_len = guess_chars.size();
        const int batch = 100000;
        for (int base = 0; base < N; base += batch) {
            int curN = std::min(batch, N - base);

            // 分批准备 offsets, lengths, out_offsets, flat_values
            std::vector<int> offsets(curN), lengths(curN), out_offsets(curN);
            std::vector<char> flat_values;
            int offset = 0, total_output_len = 0;
            for (int i = 0; i < curN; ++i) {
                const auto& s = a->ordered_values[base + i];
                offsets[i] = offset;
                lengths[i] = s.size();
                flat_values.insert(flat_values.end(), s.begin(), s.end());
                offset += s.size();
            }
            for (int i = 0; i < curN; ++i) {
                out_offsets[i] = total_output_len;
                total_output_len += guess_len + lengths[i];
            }

            // 分配并拷贝到GPU
            int *d_offsets, *d_lengths, *d_out_offsets;
            char *d_flat_values, *d_output, *d_guess;
            cudaMalloc(&d_offsets, curN * sizeof(int));
            cudaMalloc(&d_lengths, curN * sizeof(int));
            cudaMalloc(&d_out_offsets, curN * sizeof(int));
            cudaMalloc(&d_flat_values, flat_values.size() * sizeof(char));
            cudaMalloc(&d_output, total_output_len * sizeof(char));
            cudaMalloc(&d_guess, guess_len * sizeof(char));
            cudaMemcpy(d_offsets, offsets.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lengths, lengths.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_offsets, out_offsets.data(), curN * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_flat_values, flat_values.data(), flat_values.size() * sizeof(char), cudaMemcpyHostToDevice);
            cudaMemcpy(d_guess, guess_chars.data(), guess_len * sizeof(char), cudaMemcpyHostToDevice);

            // 启动kernel
            int threads = 256;
            int blocks = (curN + threads - 1) / threads;
            concat_guess<<<blocks, threads>>>(curN, d_guess, guess_len, d_flat_values, d_offsets, d_lengths, d_out_offsets, d_output);
            cudaError_t err2 = cudaGetLastError();
            if (err2 != cudaSuccess) {
                printf("CUDA kernel error: %s\n", cudaGetErrorString(err2));
                exit(1);
            }
            cudaDeviceSynchronize();

            // 拷贝回CPU
            std::vector<char> output(total_output_len);
            cudaMemcpy(output.data(), d_output, total_output_len * sizeof(char), cudaMemcpyDeviceToHost);

            // 还原字符串
            for (int i = 0; i < curN; ++i) {
                guesses.emplace_back(output.data() + out_offsets[i], guess_len + lengths[i]);
            }
            total_guesses += curN;

            // 释放GPU内存
            cudaFree(d_offsets);
            cudaFree(d_lengths);
            cudaFree(d_flat_values);
            cudaFree(d_output);
            cudaFree(d_guess);
            cudaFree(d_out_offsets);
        }
        /*
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
        */
    }
}
__global__ void batch_concat_guess(
    int N,
    const char* all_guess_chars,
    const int* pt_guess_offsets,
    const int* pt_guess_lens,
    const char* flat_values,
    const int* offsets,
    const int* lengths,
    const int* pt_id_for_value,
    const int* out_offsets,
    char* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int pt_idx = pt_id_for_value[i];
        int guess_offset = pt_guess_offsets[pt_idx];
        int guess_len = pt_guess_lens[pt_idx];
        int value_offset = offsets[i];
        int value_len = lengths[i];
        int out_start = out_offsets[i];
        // 拷贝guess
        for (int j = 0; j < guess_len; ++j)
            output[out_start + j] = all_guess_chars[guess_offset + j];
        // 拷贝value
        for (int j = 0; j < value_len; ++j)
            output[out_start + guess_len + j] = flat_values[value_offset + j];
    }
}

void PriorityQueue::BatchGenerate(const std::vector<PT>& pts, std::vector<std::string>& all_guesses) {
    // 打平所有PT的guess前缀和value
    std::vector<char> all_guess_chars;
    std::vector<int> pt_guess_offsets, pt_guess_lens;
    std::vector<char> flat_values;
    std::vector<int> offsets, lengths, pt_id_for_value;
    int total_values = 0;

    if (pts.empty()) return;

    for (int pt_idx = 0; pt_idx < pts.size(); ++pt_idx) {
        const PT& pt = pts[pt_idx];
        // 构造guess前缀
        std::string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) break;
        }
        pt_guess_offsets.push_back(all_guess_chars.size());
        pt_guess_lens.push_back(guess.size());
        all_guess_chars.insert(all_guess_chars.end(), guess.begin(), guess.end());

        // 指向最后一个segment
        segment* a = nullptr;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        if (pt.content.back().type == 3)
            a = &m.symbols[m.FindSymbol(pt.content.back())];

        if (!a) continue;
        for (const std::string& s : a->ordered_values) {
            offsets.push_back(flat_values.size());
            lengths.push_back(s.size());
            flat_values.insert(flat_values.end(), s.begin(), s.end());
            pt_id_for_value.push_back(pt_idx);
            total_values++;
        }
    }

    if (total_values == 0 || flat_values.empty() || all_guess_chars.empty()) return;

    // 计算每个输出字符串的起始位置
    std::vector<int> out_offsets(total_values);
    int total_output_len = 0;
    std::vector<int> pt_next_offset(pts.size(), 0);
    // 先统计每个PT有多少value
    std::vector<int> pt_value_count(pts.size(), 0);
    for (int i = 0; i < total_values; ++i) pt_value_count[pt_id_for_value[i]]++;
    // 计算每个PT的起始offset
    std::vector<int> pt_base_offset(pts.size(), 0);
    for (int i = 1; i < pts.size(); ++i) {
        int prev = i - 1;
        int sum = 0;
        for (int j = 0; j < pt_value_count[prev]; ++j)
            sum += pt_guess_lens[prev] + lengths[pt_next_offset[prev] + j];
        pt_base_offset[i] = pt_base_offset[i - 1] + sum;
        pt_next_offset[i] = pt_next_offset[i - 1] + pt_value_count[prev];
    }
    // 计算out_offsets
    std::vector<int> pt_cur_offset = pt_base_offset;
    for (int i = 0; i < total_values; ++i) {
        int pt_idx = pt_id_for_value[i];
        out_offsets[i] = pt_cur_offset[pt_idx];
        pt_cur_offset[pt_idx] += pt_guess_lens[pt_idx] + lengths[i];
        total_output_len += pt_guess_lens[pt_idx] + lengths[i];
    }

    // 分配并拷贝到GPU
    int *d_offsets = nullptr, *d_lengths = nullptr, *d_pt_guess_offsets = nullptr, *d_pt_guess_lens = nullptr, *d_pt_id_for_value = nullptr, *d_out_offsets = nullptr;
    char *d_flat_values = nullptr, *d_all_guess_chars = nullptr, *d_output = nullptr;
    cudaMalloc(&d_offsets, total_values * sizeof(int));
    cudaMalloc(&d_lengths, total_values * sizeof(int));
    cudaMalloc(&d_pt_guess_offsets, pts.size() * sizeof(int));
    cudaMalloc(&d_pt_guess_lens, pts.size() * sizeof(int));
    cudaMalloc(&d_pt_id_for_value, total_values * sizeof(int));
    cudaMalloc(&d_out_offsets, total_values * sizeof(int));
    cudaMalloc(&d_flat_values, flat_values.size() * sizeof(char));
    cudaMalloc(&d_all_guess_chars, all_guess_chars.size() * sizeof(char));
    cudaMalloc(&d_output, total_output_len * sizeof(char));
    cudaMemcpy(d_offsets, offsets.data(), total_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), total_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_guess_offsets, pt_guess_offsets.data(), pts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_guess_lens, pt_guess_lens.data(), pts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pt_id_for_value, pt_id_for_value.data(), total_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_offsets, out_offsets.data(), total_values * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat_values, flat_values.data(), flat_values.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_guess_chars, all_guess_chars.data(), all_guess_chars.size() * sizeof(char), cudaMemcpyHostToDevice);

    // 启动kernel
    int threads = 256;
    int blocks = (total_values + threads - 1) / threads;
    batch_concat_guess<<<blocks, threads>>>(
        total_values, d_all_guess_chars, d_pt_guess_offsets, d_pt_guess_lens,
        d_flat_values, d_offsets, d_lengths, d_pt_id_for_value, d_out_offsets, d_output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        // 释放内存
        cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_pt_guess_offsets); cudaFree(d_pt_guess_lens);
        cudaFree(d_pt_id_for_value); cudaFree(d_out_offsets); cudaFree(d_flat_values); cudaFree(d_all_guess_chars); cudaFree(d_output);
        return;
    }
    cudaDeviceSynchronize();

    // 拷贝回CPU
    std::vector<char> output(total_output_len);
    cudaMemcpy(output.data(), d_output, total_output_len * sizeof(char), cudaMemcpyDeviceToHost);

    // 还原字符串
    for (int i = 0; i < total_values; ++i) {
        int pt_idx = pt_id_for_value[i];
        int len = pt_guess_lens[pt_idx] + lengths[i];
        // 边界保护
        if (out_offsets[i] + len <= output.size())
            all_guesses.emplace_back(output.data() + out_offsets[i], len);
    }

    // 释放GPU内存
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_pt_guess_offsets);
    cudaFree(d_pt_guess_lens);
    cudaFree(d_pt_id_for_value);
    cudaFree(d_out_offsets);
    cudaFree(d_flat_values);
    cudaFree(d_all_guess_chars);
    cudaFree(d_output);
}