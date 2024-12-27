import akshare as ak
import pandas as pd
import os

def runner(year='2024'):
    """
    分析东方财富分析师指数数据，并生成 Markdown 报告。

    Args:
        year (str): 分析年份，默认为 '2024'。
    """

    # 创建输出目录
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 10 大收益最高的分析师
    df_rank = ak.stock_analyst_rank_em(year=year)
    top_10_analysts = df_rank.sort_values(by=f'{year}年收益率', ascending=False).head(10)

    markdown_output = f"# 东方财富分析师指数分析 (year={year})\n\n"
    markdown_output += "### 1. 10 大收益最高的分析师\n\n"
    markdown_output += top_10_analysts[['序号', '分析师名称', '分析师单位', f'{year}年收益率']].to_markdown(index=False) + "\n"

    # 2. 10 大收益最高的分析师的持股
    markdown_output += "\n### 2. 10 大收益最高的分析师的持股\n\n"
    top_10_stocks = pd.DataFrame()
    for index, row in top_10_analysts.iterrows():
        analyst_id = row['分析师ID']
        try:
            analyst_stocks = ak.stock_analyst_detail_em(analyst_id=analyst_id, indicator="最新跟踪成分股")
            analyst_stocks['分析师名称'] = row['分析师名称']
            top_10_stocks = pd.concat([top_10_stocks, analyst_stocks])
        except Exception as e:
            print(f"无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息: {e}")
            #markdown_output += f"\n无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息\n"

    markdown_output += top_10_stocks[['分析师名称', '股票代码', '股票名称', '当前评级名称', '最新价格', '阶段涨跌幅']].to_markdown(index=False) + "\n"

    # 3. 10 大收益最高的分析师共同看好哪些股票
    markdown_output += "\n### 3. 10 大收益最高的分析师共同看好哪些股票\n\n"
    stock_counts = top_10_stocks['股票代码'].value_counts()
    common_stocks = stock_counts[stock_counts > 1]

    if len(common_stocks) > 0:
        common_stocks_info = top_10_stocks[top_10_stocks['股票代码'].isin(common_stocks.index)][['股票代码', '股票名称']].drop_duplicates()
        markdown_output += common_stocks_info.to_markdown(index=False) + "\n"
    else:
        markdown_output += "没有发现共同看好的股票。\n"

    # 4. 10 大收益最高的分析师持股中涨幅排名前10的股票
    markdown_output += "\n### 4. 10 大收益最高的分析师持股中涨幅排名前10的股票\n\n"
    if not top_10_stocks.empty:
        top_10_best_stocks = top_10_stocks.sort_values(by='阶段涨跌幅', ascending=False).head(10)
        markdown_output += top_10_best_stocks[['股票代码', '股票名称', '阶段涨跌幅', '分析师名称']].to_markdown(index=False) + "\n"
    else:
        markdown_output += "没有可用的股票数据。\n"

    # 5. 收益最差的 10 大分析师
    bottom_10_analysts = df_rank.sort_values(by=f'{year}年收益率', ascending=True).head(10)
    markdown_output += "\n### 5. 收益最差的 10 大分析师\n\n"
    markdown_output += bottom_10_analysts[['序号', '分析师名称', '分析师单位', f'{year}年收益率']].to_markdown(index=False) + "\n"

    # 6. 收益最差的分析师持有了哪些股，这些股票的收益是怎么样的
    markdown_output += "\n### 6. 收益最差的分析师持有了哪些股，这些股票的收益是怎么样的\n\n"
    bottom_10_stocks = pd.DataFrame()
    for index, row in bottom_10_analysts.iterrows():
        analyst_id = row['分析师ID']
        try:
            analyst_stocks = ak.stock_analyst_detail_em(analyst_id=analyst_id, indicator="最新跟踪成分股")
            analyst_stocks['分析师名称'] = row['分析师名称']
            bottom_10_stocks = pd.concat([bottom_10_stocks, analyst_stocks])
        except Exception as e:
            print(f"无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息: {e}")
            #markdown_output += f"\n无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息\n"

    if not bottom_10_stocks.empty:
        markdown_output += bottom_10_stocks[['分析师名称', '股票代码', '股票名称', '当前评级名称', '最新价格', '阶段涨跌幅']].to_markdown(index=False) + "\n"
    else:
        markdown_output += "没有可用的股票数据。\n"

    # 7. 所有分析师共同看好哪些股票，这些股票的年初至今涨跌幅，以及多少分析师持有
    markdown_output += "\n### 7. 所有分析师共同看好哪些股票，这些股票的年初至今涨跌幅，以及多少分析师持有\n\n"
    all_stocks = pd.DataFrame()
    for index, row in df_rank.iterrows():
        analyst_id = row['分析师ID']
        try:
            analyst_stocks = ak.stock_analyst_detail_em(analyst_id=analyst_id, indicator="最新跟踪成分股")
            analyst_stocks['分析师名称'] = row['分析师名称']
            all_stocks = pd.concat([all_stocks, analyst_stocks])
        except Exception as e:
            print(f"无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息: {e}")
            #markdown_output += f"\n无法获取分析师 {row['分析师名称']} ({analyst_id}) 的持股信息\n"

    all_stock_counts = all_stocks['股票代码'].value_counts()
    all_common_stocks = all_stock_counts[all_stock_counts > 1]  # 出现次数大于1的股票代码

    if len(all_common_stocks) > 0:
        common_stocks_info = all_stocks[all_stocks['股票代码'].isin(all_common_stocks.index)][['股票代码', '股票名称']].drop_duplicates()
        common_stocks_info['股票代码'] = common_stocks_info['股票代码'].str.zfill(6)  # 股票代码填充0
        common_stocks_info['持有分析师数量'] = common_stocks_info['股票代码'].map(all_common_stocks) # 新增 持有分析师数量 列

        stock_spot_data = ak.stock_zh_a_spot_em()  # 获取实时行情数据，包含年初至今涨跌幅
        stock_spot_data['代码'] = stock_spot_data['代码'].str.zfill(6)

        common_stocks_with_yearly_change = pd.merge(common_stocks_info, stock_spot_data, left_on='股票代码', right_on='代码', how='inner')
        markdown_output += common_stocks_with_yearly_change[['股票代码', '股票名称', '年初至今涨跌幅', '持有分析师数量']].to_markdown(index=False) + "\n"
    else:
        markdown_output += "没有发现共同看好的股票。\n"

    # 将 Markdown 内容写入文件
    with open(f"{output_dir}/analyst_{year}.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print(f"分析报告已生成：{output_dir}/analyst_{year}.md")


if __name__ == '__main__':
    runner(year='2024')