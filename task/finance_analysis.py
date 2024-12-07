import pandas as pd
from core.utils.code_tools import code_tools
from core.utils.json_from_text import extract_json_from_text
from dealer.stock_data_provider import StockDataProvider
from core.llms.llm_factory import LLMFactory
llm_client = LLMFactory().get_instance()
stock_data_provider = StockDataProvider(llm_client)

def search_stock_code(stock_name):
    try:
        stock_code = stock_data_provider.get_stock_code(stock_name)
        if stock_code:
            return {"stock_code": stock_code}
        else:
            raise ValueError("Stock code not found for the given stock name.")
    except Exception as e:
        return {"error": str(e)}

def get_stock_info(stock_code):
    try:
        stock_info = stock_data_provider.get_stock_info(stock_code)
        if not stock_info:
            raise ValueError("No data found for the provided stock code.")
        
        return stock_info
    except Exception as e:
        return {"error": str(e)}

def get_realtime_stock_data(stock_code):
    try:
        response = stock_data_provider.get_realtime_stock_data(stock_code)
        return response
    except Exception as e:
        return {"error": str(e)}

def get_historical_daily_data(stock_code, start_date, end_date):
    try:
        historical_data = stock_data_provider.get_historical_daily_data(stock_code, start_date=start_date, end_date=end_date)

        return historical_data
    except Exception as e:
        return {"error": str(e)}

def get_stock_a_indicators(stock_code):
    try:
        stock_data = stock_data_provider.get_stock_a_indicators(stock_code)

        return stock_data
    except Exception as e:
        return {"error": str(e)}

def get_one_stock_news(stock_code, limit):
    try:
        news_list = stock_data_provider.get_one_stock_news(stock_code, limit)
        stock_news = [news['content'] for news in news_list]
        return stock_news
    except Exception as e:
        return {"error": str(e)}

def get_baidu_analysis_summary(stock_code):
    try:
        analysis_summary = stock_data_provider.get_baidu_analysis_summary(stock_code)
        return analysis_summary
    except Exception as e:
        return {"error": str(e)}

def get_stock_comments_summary(stock_code):
    try:
        raw_data = stock_data_provider.get_stock_comments_summary(stock_code)
        return raw_data
    except Exception as e:
        return {"error": str(e)}

def get_industry_pe_ratio(stock_code):
    try:
        industry_pe_ratio = stock_data_provider.get_industry_pe_ratio(symbol="证监会行业分类")
        return industry_pe_ratio
    except Exception as e:
        return {"error": str(e)}

def get_financial_analysis_summary(stock_code):
    try:
        financial_data = stock_data_provider.get_financial_analysis_summary(stock_code)
        return financial_data
    except Exception as e:
        return {"error": str(e)}

def get_main_competitors(stock_code):
    try:
        competitors_info = stock_data_provider.get_main_competitors(stock_code)
 
        return competitors_info
    except Exception as e:
        return {"error": str(e)}

def get_stock_profit_forecast(stock_code):
    try:
        profit_forecast = stock_data_provider.get_stock_profit_forecast(stock_code)
        return profit_forecast
    except Exception as e:
        return {"error": str(e)}

def LLM_analysis(stock_code, stock_info, latest_stock_data, historical_data, stock_a_indicators, stock_news, baidu_analysis_summary, stock_comments_summary, industry_pe_ratio, financial_analysis_summary, main_competitors, profit_forecast):
    try:
        combined_data = {
            'stock_code': stock_code,
            'stock_info': stock_info,
            'latest_stock_data': latest_stock_data,
            'historical_data': historical_data,
            'stock_a_indicators': stock_a_indicators,
            'stock_news': stock_news,
            'baidu_analysis_summary': baidu_analysis_summary,
            'stock_comments_summary': stock_comments_summary,
            'industry_pe_ratio': industry_pe_ratio,
            'financial_analysis_summary': financial_analysis_summary,
            'main_competitors': main_competitors,
            'profit_forecast': profit_forecast
        }
        
        prompt = f"Analyze the following stock data for {stock_code}: {combined_data}"
        response = llm_client.one_chat(prompt)
        
        if "json" in response:
            analysis_result = extract_json_from_text(response)
        else:
            analysis_result = response
        
        return analysis_result
    except Exception as e:
        return {"error": str(e)}

def collect_stock_data(stock_code, start_date="20240101", end_date="20241231"):
    """收集股票相关数据"""
    try:
        data = {
            "基本信息": get_stock_info(stock_code),
            "实时数据": get_realtime_stock_data(stock_code),
            "历史数据": get_historical_daily_data(stock_code, start_date, end_date),
            "技术指标": get_stock_a_indicators(stock_code),
            "相关新闻": get_one_stock_news(stock_code, 5),
            "百度分析": get_baidu_analysis_summary(stock_code),
            "市场评论": get_stock_comments_summary(stock_code),
            "行业PE": get_industry_pe_ratio(stock_code),
            "财务分析": get_financial_analysis_summary(stock_code),
            "主要竞争对手": get_main_competitors(stock_code),
            "盈利预测": get_stock_profit_forecast(stock_code)
        }
        return data
    except Exception as e:
        print(f"收集数据错误: {e}")
        return None

def format_data_for_llm(stock_name, stock_code, data):
    """格式化数据为LLM提示"""
    return f"""
请对 {stock_name}（{stock_code}）进行全面的投资分析，生成详细的分析报告。请包含以下方面：

1. 公司概况
- 主营业务：{data['基本信息'].get('主营业务', 'N/A')}
- 所属行业：{data['基本信息'].get('所属行业', 'N/A')}

2. 市场表现
- 当前股价：{data['实时数据'].get('最新价', 'N/A')}
- 市盈率：{data['实时数据'].get('市盈率', 'N/A')}
- 市值：{data['实时数据'].get('总市值', 'N/A')}

3. 技术面分析
{data['技术指标']}

4. 最新动态
以下是近期相关新闻：
{' '.join(data['相关新闻']) if isinstance(data['相关新闻'], list) else data['相关新闻']}

5. 行业对比
- 行业平均PE：{data['行业PE']}
- 主要竞争对手：{data['主要竞争对手']}

6. 财务分析
{data['财务分析']}

7. 投资者评论
{data['市场评论']}

8. 未来展望
{data['盈利预测']}

请基于以上信息，从以下几个方面进行分析：
1. 投资评级（买入/持有/卖出）及理由
2. 核心优势与风险
3. 未来发展前景
4. 投资建议

请用专业但易懂的语言输出分析报告。
"""

def generate_analysis_report(prompt):
    """使用LLM生成分析报告"""
    try:
        analysis = llm_client.one_chat(prompt)
        return analysis
    except Exception as e:
        print(f"生成分析报告错误: {e}")
        return None

def format_final_report(stock_name, stock_code, raw_data, analysis):
    """格式化最终报告"""
    return f"""# {stock_name}（{stock_code}）投资分析报告

## 数据来源
- 分析日期：{pd.Timestamp.now().strftime('%Y-%m-%d')}
- 数据区间：{raw_data.get('start_date', '')} 至 {raw_data.get('end_date', '')}

## 分析结果
{analysis}

## 免责声明
本报告基于公开数据分析生成，仅供参考，不构成投资建议。投资有风险，入市需谨慎。
"""

def runner(stock_name="东方精工"):
    try:
        # 1. 获取股票代码
        stock_code = search_stock_code(stock_name)
        if not stock_code:
            return None
            
        # 2. 收集数据
        print(f"正在收集 {stock_name} 的数据...")
        stock_data = collect_stock_data(stock_code)
        if not stock_data:
            return None
            
        # 3. 格式化数据为LLM提示
        print("正在生成分析提示...")
        prompt = format_data_for_llm(stock_name, stock_code, stock_data)
        
        # 4. 生成分析报告
        print("正在生成分析报告...")
        analysis = generate_analysis_report(prompt)
        if not analysis:
            return None
            
        # 5. 格式化最终报告
        print("正在格式化报告...")
        final_report = format_final_report(stock_name, stock_code, stock_data, analysis)
        
        # 6. 保存结果
        code_tools.add('stock_analysis_report', final_report)
        
        print("分析报告生成完成！")
        print("-" * 50)
        print(final_report)
        
        return final_report
        
    except Exception as e:
        print(f"运行错误: {e}")
        return None
    

if __name__ == "__main__":
    runner()