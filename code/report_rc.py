from core.utils.code_tools import code_tools

ts = code_tools["ts"]
pro = ts.pro_api()

# 获取中远海控的ts_code
tsgetter = code_tools["tsgetter"]
ts_code = tsgetter["中远海控"]

# 获取2024年以来的研报数据
start_date = '20240101'
end_date = '20241231'

# 初始化变量用于存储所有数据
all_data = []

# 分页获取数据
limit = 3000
offset = 0
while True:
    df = pro.report_rc(ts_code=ts_code, start_date=start_date, end_date=end_date, limit=limit, offset=offset)
    if df.empty:
        break
    all_data.append(df)
    offset += limit

# 合并所有数据
import pandas as pd
research_report_data = pd.concat(all_data, ignore_index=True)

# 保存结果
code_tools.add('research_report_data', research_report_data)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from core.utils.code_tools import code_tools

# 确保output文件夹存在
os.makedirs('output', exist_ok=True)

# 访问之前步骤的数据
research_report_data = code_tools['research_report_data']

# 统计研报数量
report_count = research_report_data.shape[0]

# 统计发布机构数量
org_count = research_report_data['org_name'].nunique()

# 统计评级变化
rating_changes = research_report_data['rating'].value_counts()

# 生成和保存图表
plt.figure(figsize=(10, 6))
sns.countplot(y='org_name', data=research_report_data, order=research_report_data['org_name'].value_counts().index)
plt.title('研报发布机构分布')
file_name_org = f"output/{uuid.uuid4()}.png"
plt.savefig(file_name_org)
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(y='rating', data=research_report_data, order=research_report_data['rating'].value_counts().index)
plt.title('研报评级分布')
file_name_rating = f"output/{uuid.uuid4()}.png"
plt.savefig(file_name_rating)
plt.close()
# 使用LLM API分析研报
llm_client = code_tools["llm_client"]
report_texts = research_report_data['report_title'].tolist()
analyzed_reports = []
current_batch = ""

for report in report_texts:
    if len(current_batch) + len(report) > 10000:
        response = llm_client.one_chat("分析以下研报内容: " + current_batch)
        analyzed_reports.append(response)
        current_batch = report
    else:
        current_batch += " " + report

if current_batch:
    response = llm_client.one_chat("分析以下研报内容: " + current_batch)
    analyzed_reports.append(response)

# 准备返回值
results = []
results.extend(analyzed_reports)
results.append(f"![研报发布机构分布]({file_name_org})")
results.append(f"![研报评级分布]({file_name_rating})")
results.append("主要发现：")
results.append(f"1. 研报总数: {report_count}")
results.append(f"2. 发布机构数量: {org_count}")
results.append(f"3. 评级分布: {rating_changes}")

# 将分析性的结果保存到analysis_result_2变量
analysis_result_2 = "\n".join(results)
code_tools.add("analysis_result_2", analysis_result_2)

prompt = f"""
请根据提:
{analysis_result_2}

内容,生成一份详细的markdown格式报告。报告应包含以下几个主要部分:
1. 主要发现
总结关键发现和洞察。列出最重要的3-5个发现,并用简洁的语言解释每个发现的意义。
2. 关注要点
提取主要关注点。这可能包括:

行业痛点
技术创新
政策影响
市场需求变化
等。对每个要点进行简要说明。

3. 市场竞争分析
基于analysis_result_2的内容,分析市场竞争格局:

主要竞争者及其优势
市场份额分布
竞争策略比较
潜在的市场进入者

4. 趋势和机会
根据analysis_result_2的内容,识别并描述:

行业未来发展趋势
潜在的市场机会
可能的技术突破
建议的战略方向


请确保报告结构清晰,使用适当的markdown格式(如标题、列表、强调等)来增强可读性。在每个部分中,尽可能引用analysis_result_2的具体内容来支持你的分析和结论。
"""

result = llm_client.one_chat(prompt) 

with open("output/601919.md", "w", encoding="utf-8") as f:
    f.write(result)