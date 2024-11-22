import time
from collections import defaultdict
from functools import partial, wraps
import threading
import inspect
import sys
from typing import Any
import sysconfig
import os
import site

def profile(func=None, max_depth=3, base_dir=None):
    """
    Decorator: profile the execution time of a function and its sub-functions.

    Args:
        func: function to decorate
        max_depth: maximum recursion depth
        base_dir: base directory of user's code

    Returns:
        The decorated function
    """
    if func is None:
        return partial(profile, max_depth=max_depth, base_dir=base_dir)

    if base_dir is None:
        base_module_file = func.__code__.co_filename
        base_dir = os.path.dirname(os.path.abspath(base_module_file))

    # Using thread-local storage for call stack and timing data
    thread_local = threading.local()
    
    def init_thread_local():
        """Initialize thread-local storage"""
        if not hasattr(thread_local, 'call_stack'):
            thread_local.call_stack = []
        if not hasattr(thread_local, 'timing_data'):
            thread_local.timing_data = defaultdict(
                lambda: {'total_time': 0.0, 'self_time': 0.0, 'calls': 0, 'callers': set(), 'callees': set()}
            )
        if not hasattr(thread_local, 'start_times'):
            thread_local.start_times = {}
            
    def get_func_key(frame) -> str:
        """Get a unique identifier for the function"""
        code = frame.f_code
        module = inspect.getmodule(frame)
        module_name = module.__name__ if module else 'unknown'
        return f"{module_name}.{code.co_name}"

    def should_profile_frame(frame) -> bool:
        """Determine whether to profile this frame"""
        code = frame.f_code
        module = inspect.getmodule(frame)
        module_file = getattr(module, '__file__', None)

        if module_file is None:
            # Likely a built-in module, do not profile
            return False

        module_file = os.path.abspath(module_file)

        # Exclude standard library and site-packages
        stdlib_dir = sysconfig.get_paths().get('stdlib', '')
        platstdlib_dir = sysconfig.get_paths().get('platstdlib', '')
        site_packages_dirs = site.getsitepackages()
        all_exclude_dirs = [stdlib_dir, platstdlib_dir] + site_packages_dirs

        for exclude_dir in all_exclude_dirs:
            exclude_dir = os.path.abspath(exclude_dir)
            if module_file.startswith(exclude_dir):
                return False

        # Exclude virtual environment directories
        if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
            # We are in a virtual environment
            venv_dir = os.path.abspath(sys.prefix)
            if module_file.startswith(venv_dir) and not module_file.startswith(base_dir):
                return False

        # Check if module_file is under base_dir
        base_dir_abs = os.path.abspath(base_dir)
        if module_file.startswith(base_dir_abs):
            return True

        # Otherwise, do not profile
        return False

    def trace_calls(frame, event: str, arg: Any):
        """Trace function calls"""
        if event not in ('call', 'return'):
            return trace_calls

        init_thread_local()

        if not should_profile_frame(frame):
            return trace_calls

        func_key = get_func_key(frame)

        # Handle function call
        if event == 'call':
            # Check call depth
            current_depth = len(thread_local.call_stack)
            if current_depth >= max_depth:
                return trace_calls

            # Record call relationships
            if current_depth > 0:
                caller = thread_local.call_stack[-1]
                thread_local.timing_data[caller]['callees'].add(func_key)
                thread_local.timing_data[func_key]['callers'].add(caller)

            # Record start time
            thread_local.start_times[func_key] = time.time()
            thread_local.call_stack.append(func_key)

        # Handle function return
        elif event == 'return' and func_key in thread_local.start_times:
            start_time = thread_local.start_times.pop(func_key)
            execution_time = time.time() - start_time

            timing_data = thread_local.timing_data[func_key]
            timing_data['total_time'] += execution_time
            timing_data['calls'] += 1

            # Update self execution time (exclude called functions' time)
            timing_data['self_time'] += execution_time

            if thread_local.call_stack and thread_local.call_stack[-1] == func_key:
                thread_local.call_stack.pop()

        return trace_calls

    @wraps(func)
    def _wrapped(*args, **kwargs):
        """Wrapped function to trace execution time and call relationships"""
        init_thread_local()
        old_trace = sys.gettrace()
        sys.settrace(trace_calls)

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            sys.settrace(old_trace)

            # Print profiling results
            if thread_local.timing_data:
                print("\nFunction Execution Profile:")
                print("=" * 80)

                # Sort functions by total time
                sorted_funcs = sorted(
                    thread_local.timing_data.items(),
                    key=lambda x: x[1]['total_time'],
                    reverse=True
                )

                # Calculate total execution time for percentage
                total_exec_time = sum(stats['total_time'] for func_key, stats in sorted_funcs)

                # Print detailed information for each function
                for func_key, stats in sorted_funcs:
                    module_name, func_name = func_key.rsplit('.', 1)
                    total_time = stats['total_time']
                    self_time = stats['self_time']
                    calls = stats['calls']

                    print(f"\n{func_name} [{module_name}]")
                    print("-" * 80)
                    print(f"Total time:   {total_time:10.4f}s ({total_time/total_exec_time*100:6.2f}%)")
                    print(f"Self time:    {self_time:10.4f}s ({self_time/total_exec_time*100:6.2f}%)")
                    print(f"Calls:        {calls}")
                    print(f"Time/call:    {total_time/calls:10.4f}s")

                    # Print call relationships
                    if stats['callers']:
                        print("Called by:    " + ", ".join(c.split('.')[-1] for c in stats['callers']))
                    if stats['callees']:
                        print("Calls to:     " + ", ".join(c.split('.')[-1] for c in stats['callees']))

                # Clear current statistics data
                thread_local.timing_data.clear()
                thread_local.start_times.clear()
                thread_local.call_stack.clear()

    return _wrapped