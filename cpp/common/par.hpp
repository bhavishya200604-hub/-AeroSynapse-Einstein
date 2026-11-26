// parallel_for over [0, n) on a persistent pool of Win32 threads (this compiler has neither OpenMP's
// pthread library nor std::thread). Workers sleep on a semaphore between loops; work is handed out in
// chunks through an atomic counter, and a loop ends when every woken worker has drained the counter.
// AEROSYNAPSE_THREADS sets the thread count; the default is the number of logical processors.
#pragma once
#include <cstdlib>
#include <functional>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#endif

namespace par {

inline int threads() {
    static int n = 0;
    if (n) return n;
    const char* e = getenv("AEROSYNAPSE_THREADS");
    if (e && atoi(e) > 0) n = atoi(e);
#ifdef _WIN32
    if (!n) { SYSTEM_INFO si; GetSystemInfo(&si); n = (int)si.dwNumberOfProcessors; }
#endif
    if (n < 1) n = 1;
    return n;
}

#ifdef _WIN32
// One loop at a time: a second thread that starts a loop waits for the running one (two meshing threads used to
// share the workers and corrupt each other's loops); a loop started inside a loop body runs serially.
struct Pool {
    CRITICAL_SECTION lock;
    DWORD tls = TLS_OUT_OF_INDEXES;  // set on pool workers and on the thread that is running a loop
    const std::function<void(int)>* fn = nullptr;
    volatile LONG next = 0, remaining = 0;
    int n = 0, chunk = 1, T = 1;
    HANDLE go = nullptr, done = nullptr;
    std::vector<HANDLE> th;

    void drain() {
        for (;;) {
            LONG s = InterlockedExchangeAdd(&next, chunk);
            if (s >= n) break;
            int e = (int)s + chunk < n ? (int)s + chunk : n;
            for (int i = (int)s; i < e; ++i) (*fn)(i);
        }
    }
    static unsigned __stdcall loop(void* p) {
        Pool* P = (Pool*)p;
        TlsSetValue(P->tls, (LPVOID)1);
        for (;;) {
            WaitForSingleObject(P->go, INFINITE);
            P->drain();
            if (InterlockedDecrement(&P->remaining) == 0) SetEvent(P->done);
        }
        return 0;
    }
    void start() {
        T = threads();
        if (T <= 1 || go) return;
        InitializeCriticalSection(&lock);
        tls = TlsAlloc();
        go = CreateSemaphore(nullptr, 0, T, nullptr);
        done = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        for (int t = 1; t < T; ++t) th.push_back((HANDLE)_beginthreadex(nullptr, 0, loop, this, 0, nullptr));
    }
    void run(int count, const std::function<void(int)>& f, int ch) {
        if (tls != TLS_OUT_OF_INDEXES && TlsGetValue(tls)) { for (int i = 0; i < count; ++i) f(i); return; }  // nested: serially
        EnterCriticalSection(&lock);
        TlsSetValue(tls, (LPVOID)1);
        try { run_locked(count, f, ch); }
        catch (...) { TlsSetValue(tls, nullptr); LeaveCriticalSection(&lock); throw; }
        TlsSetValue(tls, nullptr);
        LeaveCriticalSection(&lock);
    }
    void run_locked(int count, const std::function<void(int)>& f, int ch) {
        fn = &f; n = count; next = 0;
        chunk = ch > 0 ? ch : (count / (T * 8) > 0 ? count / (T * 8) : 1);
        int wake = T - 1;
        if (count / chunk < 2) { drain(); return; }  // too little work to share
        remaining = wake;
        ReleaseSemaphore(go, wake, nullptr);
        drain();
        WaitForSingleObject(done, INFINITE);
    }
};
inline Pool& pool() { static Pool p; p.start(); return p; }
#endif

inline void parallel_for(int n, const std::function<void(int)>& fn, int chunk = 0) {
    if (n <= 0) return;
    if (threads() == 1 || n == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
#ifdef _WIN32
    pool().run(n, fn, chunk);
#else
    for (int i = 0; i < n; ++i) fn(i);
#endif
}

}  // namespace par
