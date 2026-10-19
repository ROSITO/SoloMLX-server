"""Process/system memory sampling for standalone paging validation.

Darwin phys_footprint includes Metal allocations absent from ordinary RSS.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time

import psutil


class _DarwinUsage(ctypes.Structure):
    # rusage_info_v2 in the macOS SDK's sys/resource.h.
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups',
            'pageins', 'wired_size', 'resident_size', 'phys_footprint',
            'proc_start_abstime', 'proc_exit_abstime', 'child_user_time',
            'child_system_time', 'child_pkg_idle_wkups', 'child_interrupt_wkups',
            'child_pageins', 'child_elapsed_abstime', 'diskio_bytesread',
            'diskio_byteswritten')]


class _DarwinVM(ctypes.Structure):
    # task_vm_info revision 1; kernel accepts the corresponding prefix count.
    _fields_ = [('virtual_size', ctypes.c_uint64), ('region_count', ctypes.c_int32),
                ('page_size', ctypes.c_int32)] + [
        (name, ctypes.c_uint64) for name in (
            'resident_size', 'resident_size_peak', 'device', 'device_peak',
            'internal', 'internal_peak', 'external', 'external_peak',
            'reusable', 'reusable_peak', 'purgeable_volatile_pmap',
            'purgeable_volatile_resident', 'purgeable_volatile_virtual',
            'compressed', 'compressed_peak', 'compressed_lifetime', 'phys_footprint')]


class MemoryMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = []
        self.phase = 'start'
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self._libproc = None
        self._libsystem = None
        if sys.platform == 'darwin':
            self._libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
            self._libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self._libproc.proc_pid_rusage.restype = ctypes.c_int
            self._libsystem = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
            self._task = ctypes.c_uint32.in_dll(self._libsystem, 'mach_task_self_').value
            self._libsystem.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            self._libsystem.task_info.restype = ctypes.c_int

    def sample(self):
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        item = dict(monotonic=time.monotonic(), phase=self.phase,
                    rss_bytes=self._process.memory_info().rss,
                    available_bytes=vm.available, swap_used_bytes=swap.used,
                    swap_in_bytes=swap.sin, swap_out_bytes=swap.sout)
        if self._libproc is not None:
            usage = _DarwinUsage()
            if self._libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)) == 0:
                item.update(physical_footprint_bytes=usage.phys_footprint,
                            process_pageins=usage.pageins,
                            process_disk_read_bytes=usage.diskio_bytesread)
            vm_info = _DarwinVM()
            count = ctypes.c_uint32(ctypes.sizeof(vm_info) // 4)
            if self._libsystem.task_info(self._task, 22, ctypes.byref(vm_info), ctypes.byref(count)) == 0:
                item.update(process_compressed_bytes=vm_info.compressed,
                            process_compressed_peak_bytes=vm_info.compressed_peak)
        self.samples.append(item)
        return item

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, name='ssd-memory-sampler', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.sample()

    def report(self):
        samples = list(self.samples)
        if not samples:
            return {}
        physical = [s['physical_footprint_bytes'] for s in samples if 'physical_footprint_bytes' in s]
        compressed = [s['process_compressed_peak_bytes'] for s in samples if 'process_compressed_peak_bytes' in s]
        return dict(peak_rss_bytes=max(s['rss_bytes'] for s in samples),
                    peak_physical_footprint_bytes=max(physical) if physical else None,
                    peak_process_compressed_bytes=max(compressed) if compressed else None,
                    min_available_bytes=min(s['available_bytes'] for s in samples),
                    swap_growth_bytes=samples[-1]['swap_used_bytes'] - samples[0]['swap_used_bytes'],
                    samples=samples)
