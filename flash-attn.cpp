#include <torch/torch.h>
#include <iostream>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

int main()
{
    int batch_size = 16;
    int n_head = 12;
    int seq_len = 64;
    int head_embd = 64;

    auto q = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto k = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto v = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();

    forward(q, k, v);
}