from core.down_llms import down_all_files,is_need_update

if __name__ == "__main__":
    if is_need_update():
        down_all_files()