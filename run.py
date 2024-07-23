

file_path = "./code/cls_news.py"
#file_path = "./code/wordcloud.py"
def run_file(file_name,global_vars=None):
    import runpy
    runpy.run_path(file_name, init_globals= global_vars)
    

if __name__ == "__main__":
    run_file(file_path)

