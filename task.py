import sys
import os
import re
import importlib.util
import inspect
from traceback import format_exc
from typing import get_type_hints



def find_runner_files(directory):
    runner_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as file:
                        content = file.read()
                except UnicodeDecodeError:
                    print(f"Warning: Unable to read file {filename}. Skipping.")
                    continue
            
            if re.search(r'\bdef\s+runner\s*\(', content):
                runner_files[filename[:-3]] = file_path
    return runner_files

def parse_and_convert_args(func, args):
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    converted_args = []
    converted_kwargs = {}

    # 首先，填充所有默认值
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            converted_kwargs[param_name] = param.default

    # 处理传入的参数
    for i, arg in enumerate(args):
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in sig.parameters:
                param_type = type_hints.get(key, str)
                converted_kwargs[key] = safe_convert(value, param_type)
        else:
            if len(sig.parameters) > i:
                # 如果是位置参数，和参数列表中的参数对应
                param_name = list(sig.parameters.keys())[i]
                param_type = type_hints.get(param_name, str)
                if arg:  # 只有当参数非空时才转换
                    converted_kwargs[param_name] = safe_convert(arg, param_type)

    return [], converted_kwargs  # 返回空的 args 列表和填充的 kwargs 字典

def safe_convert(value, param_type):
    if value == '':
        return None  # 返回 None 而不是 0，让函数使用其默认值
    try:
        return param_type(value)
    except ValueError:
        print(f"Warning: Could not convert '{value}' to {param_type.__name__}. Using original value.")
        return value

def load_and_run(file_path, args):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, 'runner'):
        runner_func = module.runner
        _, converted_kwargs = parse_and_convert_args(runner_func, args)
        try:
            runner_func(**converted_kwargs)
        except Exception as e:
            print(f"Error executing runner in {file_path}: {str(e)} \n {format_exc()}")
            print(f"Kwargs: {converted_kwargs}")
    else:
        print(f"Error: 'runner' function not found in {file_path}")

def main():
    task_directory = './task'
    runner_files = find_runner_files(task_directory)

    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in runner_files:
            args = sys.argv[2:]
            print(f"function name: {func_name}")
            print(f"args: {args}")
            
            load_and_run(runner_files[func_name], args)
        else:
            print(f"Error: Function '{func_name}' not found.")
    else:
        print("Usage: python task.py <function_name> [args]")

if __name__ == "__main__":
    main()