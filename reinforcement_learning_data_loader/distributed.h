#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

// NOTE: ProcessGroupNCCL / Gloo headers only needed if USE_DISTRIBUTED is defined.
// The standard libtorch binary does not ship ProcessGroupGloo; leave it out.
#if defined(USE_DISTRIBUTED)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#endif

namespace dl
{

struct DistributedContext
{
    int rank       = 0;
    int world_size = 1;
    int local_rank = 0;
    torch::Device device = torch::kCPU;
    bool is_distributed() const { return world_size > 1; }
};

// FIX: guard the static variable declaration inside USE_DISTRIBUTED too,
// so it doesn't fail to compile when the distributed headers are absent.
#if defined(USE_DISTRIBUTED)
// Use the concrete NCCL type instead of the removed abstract ProcessGroup base.
static c10::intrusive_ptr<c10d::ProcessGroupNCCL> g_process_group;
#endif

inline DistributedContext init_distributed(const std::string& backend = "nccl")
{
    const char* rank_env = std::getenv("RANK");

    // ── Single-process fallback ───────────────────────────────
    if (!rank_env)
    {
        DistributedContext ctx;
        ctx.rank       = 0;
        ctx.world_size = 1;
        ctx.local_rank = 0;
        ctx.device     = torch::cuda::is_available()
                         ? torch::Device(torch::kCUDA, 0)
                         : torch::Device(torch::kCPU);
        return ctx;
    }

    DistributedContext ctx;
    ctx.rank       = std::stoi(rank_env);
    const char* ws_env = std::getenv("WORLD_SIZE");
    ctx.world_size = ws_env ? std::stoi(ws_env) : 1;
    const char* lr_env = std::getenv("LOCAL_RANK");
    ctx.local_rank = lr_env ? std::stoi(lr_env) : ctx.rank;

    if (torch::cuda::is_available())
    {
        ctx.device = torch::Device(torch::kCUDA, ctx.local_rank);
        std::cout << "[distributed] rank=" << ctx.rank
                  << " local_rank=" << ctx.local_rank
                  << " world_size=" << ctx.world_size
                  << " device=cuda:" << ctx.local_rank << "\n";
    }
    else
    {
        ctx.device = torch::Device(torch::kCPU);
        std::cout << "[distributed] CPU mode rank=" << ctx.rank << "\n";
    }

#if defined(USE_DISTRIBUTED)
    const char* master_addr = std::getenv("MASTER_ADDR");
    const char* master_port = std::getenv("MASTER_PORT");
    if (!master_addr || !master_port)
        throw std::runtime_error("MASTER_ADDR / MASTER_PORT not set for distributed init");

    // FIX: TCPStore must be wrapped in intrusive_ptr, not shared_ptr
    auto store = c10::make_intrusive<c10d::TCPStore>(
        std::string(master_addr),
        c10d::TCPStoreOptions{
            static_cast<uint16_t>(std::stoi(master_port)),
            ctx.rank == 0,
            static_cast<size_t>(ctx.world_size)
        });

    auto opts = c10d::ProcessGroupNCCL::Options::create();
    g_process_group = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store, ctx.rank, ctx.world_size, opts);
#endif

    return ctx;
}

inline void cleanup_distributed()
{
#if defined(USE_DISTRIBUTED)
    g_process_group.reset();
    if (torch::cuda::is_available())
        c10::cuda::device_synchronize();
#endif
}

inline void barrier(const DistributedContext& ctx)
{
    if (ctx.world_size <= 1) return;
#if defined(USE_DISTRIBUTED)
    if (g_process_group)
    {
        auto work = g_process_group->barrier();
        work->wait();
    }
#endif
}

} // namespace dl