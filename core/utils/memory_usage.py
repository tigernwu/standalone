import psutil
import os
import wmi

def get_top_memory_consuming_processes(num_processes=5):
  """
  获取内存占用前 num_processes 的进程信息.

  Args:
    num_processes: 要获取的进程数量.

  Returns:
    一个包含进程信息（名称、内存占用）的列表.
  """

  processes = []
  for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    try:
      process_info = proc.info
      processes.append({
          'pid': process_info['pid'],
          'name': process_info['name'],
          'memory_usage': process_info['memory_info'].rss / (1024 * 1024)  # 以 MB 为单位
      })
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass

  # 按内存占用排序
  processes.sort(key=lambda proc: proc['memory_usage'], reverse=True)

  return processes[:num_processes]

def get_total_memory_usage_all():
    """Calculates the total memory usage of all processes running on the system."""

    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)  # Convert bytes to GB


def get_total_memory_usage_linux():
    """
    Calculates the total memory usage of all processes running on the system.
    """
    total_memory_usage = 0
    for pid_str in os.listdir('/proc'):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            with open(f'/proc/{pid}/statm') as f:
                rss_pages = int(f.readline().split()[1])
                total_memory_usage += rss_pages * 4096  # Assuming page size is 4KB
        except FileNotFoundError:
            pass  # Process might have terminated while iterating
    return total_memory_usage / (1024 ** 3)  # Convert bytes to GB

def get_total_memory_usage_windows():
    """
    Calculates the total memory usage of all processes running on a Windows system using WMI.
    """
    c = wmi.WMI()
    total_memory_usage = 0
    for process in c.Win32_Process():
        total_memory_usage += int(process.WorkingSetSize)
    return total_memory_usage / (1024 ** 3)  # Convert bytes to GB

if __name__ == "__main__":
  # top_processes = get_top_memory_consuming_processes()
  # for proc in top_processes:
  #   print(f"进程名称: {proc['name']}, 进程ID: {proc['pid']}, 内存占用: {proc['memory_usage']:.2f} MB")
    total_memory_gb = get_total_memory_usage_windows()
    print(f"Total memory usage: {total_memory_gb:.2f} GB")