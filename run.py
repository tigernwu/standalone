
from core.utils.timer import timer
from core.utils.code_tools_required import add_required_tools
file_path = "./code/cls_news.py"
#file_path = "./code/llm_predict.py"
#file_path = "./code/wordcloud.py"
#file_path = "./code/corr_fin_futures.py"
#file_path = "./code/report_rc.py"
@timer
def run_file(file_name,global_vars=None):
    import runpy
    runpy.run_path(file_name, init_globals= global_vars)
    
if __name__ == "__main__":
    run_file(file_path)

