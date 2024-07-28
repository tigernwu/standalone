import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from collections import Counter
from wordcloud import WordCloud
import jieba
import re
from core.utils.config_setting import Config
from core.llms.llm_factory import LLMFactory
from datetime import datetime

cls_telegraph_news = ak.stock_info_global_cls()

# 将发布日期和发布时间组合成新的列日期
cls_telegraph_news["日期"] = cls_telegraph_news.apply(
    lambda row: datetime.combine(row["发布日期"], row["发布时间"]), axis=1
)

# 获取起始时间和结束时间
start_time = cls_telegraph_news["日期"].min()
end_time = cls_telegraph_news["日期"].max()

# 调整起始时间和结束时间
if start_time.minute < 30:
    start_time = start_time.replace(minute=0, second=0, microsecond=0)
else:
    start_time = start_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(
        hours=1
    )

if end_time.minute < 30:
    end_time = end_time.replace(minute=0, second=0, microsecond=0)
else:
    end_time = end_time.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(
        hours=1
    )

# 生成time_span字符串
time_span = f"资讯时间从{start_time.strftime('%Y年%m月%d日%H点')}，到{end_time.strftime('%Y年%m月%d日%H点')}"
# config = Config()
llm_factory = LLMFactory()

# 确保output文件夹存在
os.makedirs("output", exist_ok=True)

# 设置字体路径
if os.name == "nt":
    font_path = "C:/Windows/Fonts/msyh.ttc"
elif os.name == "posix":
    if os.path.exists("/System/Library/Fonts/PingFang.ttc"):
        font_path = "/System/Library/Fonts/PingFang.ttc"
    else:
        font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
else:
    font_path = None

# 设置matplotlib的字体
plt.rcParams["font.sans-serif"] = [
    font_path,
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "sans-serif",
]

# 访问之前步骤的数据
data = cls_telegraph_news
df = data

# 访问之前步骤的数据
data = cls_telegraph_news
df = data


# 1. 词云图
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = jieba.cut(text)
    stop_words = llm_factory.stop_words
    return [word for word in words if word not in stop_words and len(word) > 1]


all_words = []
for _, row in df.iterrows():
    all_words.extend(preprocess_text(row["标题"]))

word_freq = Counter(all_words)
filtered_words = {word: freq for word, freq in word_freq.items() if freq <= 120}

wordcloud = WordCloud(
    width=800, height=400, background_color="white", font_path=font_path, max_words=120
).generate_from_frequencies(filtered_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
wordcloud_file = f"output/{uuid.uuid4()}.png"
plt.savefig(wordcloud_file)
plt.close()

# 2. 使用LLM API进行新闻分析
llm_client = llm_factory.get_instance()

news_batches = []
current_batch = ""
for _, row in data.iterrows():
    news = f"标题: {row['标题']}\n内容: {row['内容']}\n"
    if len(current_batch) + len(news) > 9500:
        news_batches.append(current_batch)
        current_batch = news
    else:
        current_batch += news
if current_batch:
    news_batches.append(current_batch)

analysis_results = []
for batch in news_batches:
    prompt = f"""分析以下新闻，重点关注：
    1. 总结和提炼对市场影响比较大的内容
    2. 金融市场动态总结
    3. 市场情绪的走向和判断
    4. 市场影响、热点和异常
    5. 行业影响、热点和异常
    6. 其他的市场重点要点信息

    新闻内容：
    {batch}
    """
    response = llm_client.one_chat(prompt)
    analysis_results.append(response)

# 准备返回值
results = []
results.append(f"![新闻标题词云图]({wordcloud_file})")
results.append(time_span)
results.append("新闻分析结果：")
for i, result in enumerate(analysis_results, 1):
    results.append(f"批次 {i} 分析：\n{result}\n")

# 将结果保存到analysis_result变量
analysis_result = "\n".join(results)
reporter = llm_factory.get_reporter()
# 使用LLM API生成markdown文件
prompt = f"""。
请将以下新闻分析结果整理成markdown格式的报告，并生成一个包含新闻标题词云图的markdown文件.
报告主要内容及markdown文件的结构如下：
标题:xxxx年xx月xx日财经资讯分析报告
(本次分析的时间范围：xxxx年xx月xx日xx点至xxxx年xx月xx日xx点)

1. 主要市场趋势: 
   - 综合所有批次,识别出最重要、最具影响力的市场趋势。
   - 这些趋势如何相互关联或冲突?

2. 金融市场动态:
   - 总结各个市场(如股票、债券、商品等)的整体表现和关键变动。
   - 识别出可能影响未来市场走向的关键因素。

3. 市场情绪分析:
   - 综合评估整体市场情绪(如乐观、悲观、谨慎等)。
   - 分析情绪变化的原因和可能的影响。

4. 热点和异常事件:
   - 列出所有批次中提到的主要热点和异常事件。
   - 评估这些事件对市场的短期和长期影响。

5. 行业分析:
   - 识别受关注度最高或影响最大的行业。
   - 总结这些行业面临的机遇和挑战。

6. 政策和监管影响:
   - 总结可能影响市场的主要政策或监管变化。
   - 分析这些变化可能带来的影响。

7. 风险评估:
   - 基于所有分析结果,识别潜在的系统性风险或值得关注的风险因素。

8. 前瞻性展望:
   - 根据当前分析,对短期和中期市场走势做出预测。
   - 提出投资者和市场参与者应该关注的关键点。

9. 词云分析:
   - 结合新闻标题词云图,分析高频词与市场趋势的关联。
   - 探讨词云中反映出的市场焦点与实际分析结果的一致性。

请提供一个全面、深入、结构化的总结,整合所有批次的分析结果,突出最重要的发现和见解。

需要总结的新闻内容摘要如下：
{analysis_result}
"""

prompt1 = f"""
请根据以下多批次的新闻摘要提供一个全面、深入、结构化的总结,整合所有批次的分析结果,突出最重要的发现和见解。包括标题、时间范围、主要市场趋势、金融市场动态、市场情绪分析、热点和异常事件、行业分析、政策和监管影响、风险评估、前瞻性展望和词云分析。请确保分析结果准确、全面，并提供详细的解释和分析。以markdown格式输出。

需要总结的新闻内容摘要如下：
{analysis_result}
"""
response = reporter.one_chat(prompt)
date_str = datetime.now().strftime("%Y%m%d%H%M%S")
file = f"markdowns/news/news{date_str}.md"
# 创建或更新output.md文件
with open(file, "w", encoding="utf-8") as f:
    f.write(response)