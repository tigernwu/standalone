import akshare as ak
import pandas as pd
import os
from datetime import datetime

def get_stock_rank_cxg_ths(symbol="创月新高"):
    """创新高"""
    try:
        df = ak.stock_rank_cxg_ths(symbol=symbol)
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_cxg_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_cxd_ths(symbol="创月新低"):
    """创新低"""
    try:
        df = ak.stock_rank_cxd_ths(symbol=symbol)
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_cxd_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_lxsz_ths():
    """连续上涨"""
    try:
        df = ak.stock_rank_lxsz_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_lxsz_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_lxxd_ths():
    """连续下跌"""
    try:
        df = ak.stock_rank_lxxd_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_lxxd_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_cxfl_ths():
    """持续放量"""
    try:
        df = ak.stock_rank_cxfl_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_cxfl_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_cxsl_ths():
    """持续缩量"""
    try:
        df = ak.stock_rank_cxsl_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_cxsl_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_xstp_ths(symbol="500日均线"):
    """向上突破"""
    try:
        df = ak.stock_rank_xstp_ths(symbol=symbol)
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_xstp_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_xxtp_ths(symbol="500日均线"):
    """向下突破"""
    try:
        df = ak.stock_rank_xxtp_ths(symbol=symbol)
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_xxtp_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_ljqs_ths():
    """量价齐升"""
    try:
        df = ak.stock_rank_ljqs_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_ljqs_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_ljqd_ths():
    """量价齐跌"""
    try:
        df = ak.stock_rank_ljqd_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_ljqd_ths: {e}")
        return pd.DataFrame()

def get_stock_rank_xzjp_ths():
    """险资举牌"""
    try:
        df = ak.stock_rank_xzjp_ths()
        return df
    except Exception as e:
        print(f"Error in get_stock_rank_xzjp_ths: {e}")
        return pd.DataFrame()

def save_to_markdown(df, title, file):
    """将 DataFrame 保存为 Markdown 表格"""
    if not df.empty:
        markdown_table = df.to_markdown(index=False)
        file.write(f"## {title}\n\n")
        file.write(markdown_table)
        file.write("\n\n")

def get_intersection(df1, df2, title, file):
    """获取两个 DataFrame 的交集并保存到 Markdown 文件"""
    if df1.empty or df2.empty:
      file.write(f"## {title}\n\n")
      file.write("无数据\n\n")
      return
    
    intersection = pd.merge(df1, df2, on=["股票代码", "股票简称"], how="inner")
    if not intersection.empty:
        save_to_markdown(intersection, title, file)
    else:
        file.write(f"## {title}\n\n")
        file.write("无共同股票\n\n")

def runner():
    """主函数"""
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    today_date = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(output_dir, f"stock_rank_{today_date}.md")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# {today_date} 股票数据分析\n\n")

        # 创新高
        for symbol in ["创月新高", "半年新高", "一年新高", "历史新高"]:
            df = get_stock_rank_cxg_ths(symbol)
            save_to_markdown(df, f"创新高-{symbol}", f)

        # 创新低
        for symbol in ["创月新低", "半年新低", "一年新低", "历史新低"]:
            df = get_stock_rank_cxd_ths(symbol)
            save_to_markdown(df, f"创新低-{symbol}", f)

        # 连续上涨
        df_lxsz = get_stock_rank_lxsz_ths()
        save_to_markdown(df_lxsz, "连续上涨", f)

        # 连续下跌
        df_lxxd = get_stock_rank_lxxd_ths()
        save_to_markdown(df_lxxd, "连续下跌", f)

        # 持续放量
        df_cxfl = get_stock_rank_cxfl_ths()
        save_to_markdown(df_cxfl, "持续放量", f)

        # 持续缩量
        df_cxsl = get_stock_rank_cxsl_ths()
        save_to_markdown(df_cxsl, "持续缩量", f)

        # 向上突破
        for symbol in ["20日均线", "30日均线", "60日均线", "90日均线"]:
            df = get_stock_rank_xstp_ths(symbol)
            save_to_markdown(df, f"向上突破-{symbol}", f)

        # 向下突破
        for symbol in ["20日均线", "30日均线", "60日均线", "90日均线"]:
            df = get_stock_rank_xxtp_ths(symbol)
            save_to_markdown(df, f"向下突破-{symbol}", f)

        # 量价齐升
        df_ljqs = get_stock_rank_ljqs_ths()
        save_to_markdown(df_ljqs, "量价齐升", f)

        # 量价齐跌
        df_ljqd = get_stock_rank_ljqd_ths()
        save_to_markdown(df_ljqd, "量价齐跌", f)

        # 险资举牌
        df_xzjp = get_stock_rank_xzjp_ths()
        save_to_markdown(df_xzjp, "险资举牌", f)

        # 持续放量和持续上涨共同的股票
        get_intersection(df_cxfl, df_lxsz, "持续放量和持续上涨共同的股票", f)

        # 持续缩量和持续下跌的共同股票
        get_intersection(df_cxsl, df_lxxd, "持续缩量和持续下跌的共同股票", f)

        # 持续放量和突破月新高共同的股票
        df_cxyxg = get_stock_rank_cxg_ths("创月新高")
        get_intersection(df_cxfl, df_cxyxg, "持续放量和突破月新高共同的股票", f)

        # 持续缩量和突破月新低共同的股票
        df_cxyxd = get_stock_rank_cxd_ths("创月新低")
        get_intersection(df_cxsl, df_cxyxd, "持续缩量和突破月新低共同的股票", f)

        # 持续放量和向上突破共同的股票
        for symbol in [ "20日均线", "30日均线", "60日均线"]:
          df_xstp = get_stock_rank_xstp_ths(symbol)
          get_intersection(df_cxfl, df_xstp, f"持续放量和向上突破{symbol}共同的股票", f)
        # 持续缩量和向下突破共同的股票
        for symbol in ["20日均线", "30日均线", "60日均线" ]:
            df_xxtp = get_stock_rank_xxtp_ths(symbol)
            get_intersection(df_cxsl, df_xxtp, f"持续缩量和向下突破{symbol}共同的股票", f)

if __name__ == "__main__":
    runner()