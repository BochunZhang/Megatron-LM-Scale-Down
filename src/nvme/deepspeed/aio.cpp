// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from DeepSpeed csrc/aio at c7cc64a90162214dbb8f0b29fc1034e6075e6d23.
// The rolling libaio queue and worker are specialized for FSDP host swap buffers.

#include <Python.h>

#include <fcntl.h>
#include <libaio.h>
#include <nvtx3/nvtx3.hpp>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

namespace {

constexpr size_t kAlignment = 4096;
constexpr size_t kBlockSize = 128 * 1024;
constexpr int kQueueDepth = 8;
constexpr char kWorkerCapsule[] = "megatron_nvme_aio.worker";

[[noreturn]] void fail(int error, const char* operation, const std::string& path)
{
    throw std::system_error(error, std::generic_category(), std::string(operation) + " " + path);
}

struct File {
    const int fd;
    ~File() { close(fd); }
};

class AioContext {
public:
    ~AioContext() { reset(); }

    io_context_t get(const std::string& path)
    {
        if (!context_) {
            const int result = io_setup(kQueueDepth, &context_);
            if (result < 0) { fail(-result, "io_setup", path); }
        }
        return context_;
    }

    void reset() noexcept
    {
        if (!context_) { return; }
        int result;
        do {
            result = io_destroy(context_);
        } while (result == -EINTR);
        // io_destroy must join outstanding DMA before returning the buffer to Python.
        if (result != 0) { std::terminate(); }
        context_ = nullptr;
    }

private:
    io_context_t context_ = nullptr;
};

class AioWorker {
public:
    AioWorker() : thread_([this] { run(); }) {}

    ~AioWorker()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        thread_.join();
    }

    void transfer(char* buffer, size_t size, std::string path, bool write)
    {
        std::packaged_task<void()> task([this, buffer, size, path = std::move(path), write] {
            transfer_on_worker(buffer, size, path, write);
        });
        auto done = task.get_future();
        {
            std::lock_guard lock(mutex_);
            requests_.push(std::move(task));
        }
        ready_.notify_one();
        done.get();
    }

private:
    void run()
    {
        pthread_setname_np(pthread_self(), "mfsdp-nvme-aio");
        while (true) {
            std::packaged_task<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !requests_.empty(); });
                if (requests_.empty()) { return; }
                task = std::move(requests_.front());
                requests_.pop();
            }
            task();
        }
    }

    void transfer_on_worker(char* buffer, size_t size, const std::string& path, bool write)
    {
        const auto slash = path.find_last_of('/');
        const auto name = path.substr(slash == std::string::npos ? 0 : slash + 1);
        const auto label = (write ? "NVMe AIO write " : "NVMe AIO read ") + name +
                           " bytes=" + std::to_string(size);
        const nvtx3::scoped_range range{label.c_str()};

        const int fd = open(path.c_str(), O_DIRECT | O_CLOEXEC | (write ? O_CREAT | O_WRONLY : O_RDONLY),
                            0600);
        if (fd < 0) { fail(errno, "open", path); }
        const File file{fd};
        const auto context = context_.get(path);

        std::array<iocb, kQueueDepth> requests{};
        std::array<io_event, kQueueDepth> events{};
        std::array<size_t, kQueueDepth> in_flight{};
        size_t offset = 0;
        int pending = 0;
        try {
            while (offset < size || pending > 0) {
                // Refill the queue while previously submitted blocks are still in flight.
                std::array<iocb*, kQueueDepth> batch{};
                int count = 0;
                size_t next_offset = offset;
                for (int slot = 0; slot < kQueueDepth && next_offset < size; ++slot) {
                    if (in_flight[slot]) { continue; }
                    auto* request = &requests[slot];
                    const size_t bytes = std::min(kBlockSize, size - next_offset);
                    if (write) {
                        io_prep_pwrite(request, file.fd, buffer + next_offset, bytes, next_offset);
                    } else {
                        io_prep_pread(request, file.fd, buffer + next_offset, bytes, next_offset);
                    }
                    batch[count++] = request;
                    next_offset += bytes;
                }
                if (count > 0) {
                    int submitted;
                    {
                        const nvtx3::scoped_range submit_range{"NVMe AIO io_submit"};
                        do {
                            submitted = io_submit(context, count, batch.data());
                        } while (submitted == -EINTR);
                    }
                    if (submitted < 0 && (submitted != -EAGAIN || pending == 0)) {
                        fail(-submitted, "io_submit", path);
                    }
                    if (submitted == 0 && pending == 0) { fail(EIO, "io_submit", path); }
                    for (int i = 0; i < submitted; ++i) {
                        const size_t bytes = std::min(kBlockSize, size - offset);
                        in_flight[batch[i] - requests.data()] = bytes;
                        offset += bytes;
                        ++pending;
                    }
                }

                int completed;
                {
                    const nvtx3::scoped_range wait_range{"NVMe AIO io_getevents"};
                    do {
                        completed = io_getevents(context, 1, pending, events.data(), nullptr);
                    } while (completed == -EINTR);
                }
                if (completed <= 0) {
                    fail(completed < 0 ? -completed : EIO, "io_getevents", path);
                }
                pending -= completed;
                for (int i = 0; i < completed; ++i) {
                    const auto& event = events[i];
                    const auto slot = event.obj - requests.data();
                    const auto result = static_cast<long>(event.res);
                    if (result < 0) { fail(-result, write ? "pwrite" : "pread", path); }
                    if (event.res2 != 0 || static_cast<size_t>(result) != in_flight[slot]) {
                        fail(EIO, write ? "short pwrite" : "short pread", path);
                    }
                    in_flight[slot] = 0;
                }
            }
        } catch (...) {
            // Reap or cancel accepted I/O before the file, descriptors and buffer can go away.
            context_.reset();
            throw;
        }
    }

    AioContext context_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::packaged_task<void()>> requests_;
    bool stopping_ = false;
    std::thread thread_;
};

class ReleaseGIL {
public:
    ReleaseGIL() : state_(PyEval_SaveThread()) {}
    ~ReleaseGIL() { PyEval_RestoreThread(state_); }

private:
    PyThreadState* const state_;
};

void delete_worker(PyObject* capsule)
{
    delete static_cast<AioWorker*>(PyCapsule_GetPointer(capsule, kWorkerCapsule));
}

PyObject* create_worker(PyObject*, PyObject*)
{
    try {
        auto worker = std::make_unique<AioWorker>();
        auto* capsule = PyCapsule_New(worker.get(), kWorkerCapsule, delete_worker);
        if (capsule) { worker.release(); }
        return capsule;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject* transfer(PyObject*, PyObject* args)
{
    PyObject* capsule;
    Py_buffer view{};
    const char* path;
    int write;
    if (!PyArg_ParseTuple(args, "Oy*sp", &capsule, &view, &path, &write)) { return nullptr; }
    auto* worker = static_cast<AioWorker*>(PyCapsule_GetPointer(capsule, kWorkerCapsule));
    if (!worker) {
        PyBuffer_Release(&view);
        return nullptr;
    }
    if (view.len <= 0 || view.len % kAlignment ||
        reinterpret_cast<uintptr_t>(view.buf) % kAlignment || (!write && view.readonly)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "AIO needs a 4096-byte-aligned buffer and a writable read destination");
        return nullptr;
    }

    try {
        const ReleaseGIL release;
        worker->transfer(static_cast<char*>(view.buf), view.len, path, write);
    } catch (const std::system_error& error) {
        PyErr_SetString(PyExc_OSError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    PyBuffer_Release(&view);
    if (PyErr_Occurred()) { return nullptr; }
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"create_worker", create_worker, METH_NOARGS, "Start a native libaio worker."},
    {"transfer", transfer, METH_VARARGS, "Transfer an aligned buffer and wait for completion."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "megatron_nvme_aio", "Megatron-FSDP native NVMe AIO.", -1, methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_megatron_nvme_aio() { return PyModule_Create(&module); }
