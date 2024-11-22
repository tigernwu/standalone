import sys
import os
from importlib import resources
import warnings
from typing import Union
from contextlib import contextmanager
import importlib.util

def get_python_version():
    """获取 Python 主版本和次版本号"""
    return tuple(map(int, sys.version.split('.')[:2]))

@contextmanager
def new_path_implementation(package, resource):
    """
    新的 path 实现，兼容 Python 3.13
    """
    if isinstance(package, str):
        package_path = package.replace('.', os.sep)
    else:
        package_path = package.__name__.replace('.', os.sep)
    
    # 获取包的路径
    if isinstance(package, str):
        spec = importlib.util.find_spec(package)
        if spec is None:
            raise ImportError(f"No module named '{package}'")
        package_location = os.path.dirname(spec.origin)
    else:
        package_location = os.path.dirname(package.__file__)
    
    # 构建资源文件的完整路径
    resource_path = os.path.join(package_location, resource)
    
    try:
        yield resource_path
    finally:
        pass

def patch_akshare():
    """
    为 AKShare 创建 Python 3.13 兼容性补丁
    """
    python_version = get_python_version()
    if python_version < (3, 13):
        return True

    try:
        import akshare.datasets as datasets
        import akshare.stock_feature.stock_fund_flow as stock_fund_flow
        
        # 直接替换 get_ths_js 函数
        def new_get_ths_js(file: str) -> str:
            """
            新的 get_ths_js 实现
            """
            try:
                module_path = os.path.dirname(datasets.__file__)
                data_path = os.path.join(module_path, "data", file)
                
                if os.path.exists(data_path):
                    return data_path
                else:
                    raise FileNotFoundError(f"File not found: {data_path}")
            except Exception as e:
                warnings.warn(f"获取 ths.js 文件路径时出错: {str(e)}")
                return os.path.join(os.path.dirname(datasets.__file__), "data", file)
        
        # 直接替换 _get_file_content_ths 函数
        def new_get_file_content_ths(file: str) -> str:
            """
            新的 _get_file_content_ths 实现
            """
            try:
                file_path = new_get_ths_js(file)
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                warnings.warn(f"读取文件内容时出错: {str(e)}")
                return ""
        
        # 应用补丁
        datasets.get_ths_js = new_get_ths_js
        stock_fund_flow._get_file_content_ths = new_get_file_content_ths
        
        # 如果有其他使用 resources.path 的地方，我们提供一个兼容的实现
        import importlib.resources as resources
        resources.path = new_path_implementation
        
        return True
    except Exception as e:
        warnings.warn(f"应用补丁时出错: {str(e)}")
        return False

def check_environment():
    """
    检查当前环境，返回详细信息
    """
    python_version = get_python_version()
    try:
        import akshare
        akshare_version = getattr(akshare, "__version__", "未知")
    except ImportError:
        akshare_version = "未安装"
    
    return {
        "python_version": f"{python_version[0]}.{python_version[1]}",
        "akshare_version": akshare_version,
        "needs_patch": python_version >= (3, 13)
    }

# 使用示例
if __name__ == "__main__":
    env_info = check_environment()
    print(f"环境信息:")
    print(f"Python 版本: {env_info['python_version']}")
    print(f"AKShare 版本: {env_info['akshare_version']}")
    print(f"需要补丁: {'是' if env_info['needs_patch'] else '否'}")
    
    if env_info['needs_patch']:
        success = patch_akshare()
        print(f"补丁应用{'成功' if success else '失败'}")
    else:
        print("当前 Python 版本无需应用补丁")