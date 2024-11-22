from typing import List

from core.utils.query_executor import QueryExecutor


class MarketNewsAnalyzer:
    def __init__(self):
        self.base_prompt_template = """
【分析目标】
从以下市场新闻中提取关键信息并进行深度分析。

【新闻内容】
{news_content}

【分析维度】
1. 政策动向分析
   - 新政策发布和政策信号
   - 可能带来的投资机会
   - 政策影响的行业范围

2. 行业热点跟踪
   - 重大事件和突破进展
   - 产业链上下游变化
   - 技术创新和突破

3. 资金流向分析
   - 主力资金流向板块
   - 热点板块轮动情况

4. 市场情绪指标
   - 整体市场情绪
   - 重点关注领域
   - 风险提示因素

5. 热点持续性评估
   - 前期热点延续性
   - 新热点形成原因
   - 热点转换趋势

【输出要求】
{output_requirements}
"""

    def create_initial_summary_prompt(self, max_words=500):
        """创建初始摘要的提示词模板"""
        return self.base_prompt_template.format(
            news_content="{news_content}",
            output_requirements=f"""
请输出以下JSON格式的分析结果（总字数控制在{max_words}字以内）：
```json
{{
    "policy_insights": [
        {{"topic": "政策主题", "impact": "影响分析", "opportunities": ["机会1", "机会2"]}}
    ],
    "industry_highlights": [
        {{"sector": "行业", "event": "重大事件", "significance": "重要性分析"}}
    ],
    "capital_flows": [
        {{"sector": "板块", "direction": "流向", "amount": "资金量级"}}
    ],
    "market_sentiment": {{
        "overall_mood": "整体情绪",
        "focus_areas": ["关注点1", "关注点2"],
        "risk_factors": ["风险1", "风险2"]
    }},
    "trend_analysis": {{
        "continuing_trends": ["持续热点1", "持续热点2"],
        "emerging_trends": ["新兴热点1", "新兴热点2"],
        "fading_trends": ["消退热点1", "消退热点2"]
    }}
}}
```
"""
        )

    def create_merge_summary_prompt(self, max_words=500):
        """创建合并多个摘要的提示词模板"""
        return """
【任务说明】
合并以下多个市场分析结果，提炼核心信息。

【分析结果集】
{summaries}

【合并要求】
1. 保留最重要和最新的信息
2. 去除重复的观点和数据
3. 确保信息的连贯性和完整性
4. 总字数控制在{max_words}字以内

【输出格式】
与原JSON格式保持一致
"""

    def create_compression_prompt(self, max_words=500):
        """创建压缩摘要的提示词模板"""
        return """
【压缩任务】
在保留核心信息的前提下，将以下市场分析压缩到{max_words}字以内。

【原始分析】
{content}

【压缩要求】
1. 优先保留：
   - 最新的政策信号
   - 最重要的行业事件
   - 明显的资金流向
   - 关键的市场情绪指标
2. 可以省略：
   - 重复的信息
   - 次要的细节
   - 较早的历史数据
3. 保持原有JSON格式

【输出格式】
与原JSON格式保持一致
"""

    async def analyze_news(self, news_chunks: List[str], query_executor: QueryExecutor) -> str:
        """分析新闻的主方法"""
        # 1. 初始摘要阶段
        initial_prompt = self.create_initial_summary_prompt()
        initial_summaries = await query_executor.concurrent_query(
            [initial_prompt.format(news_content=chunk) for chunk in news_chunks]
        )

        # 2. 合并摘要阶段
        if len(initial_summaries) > 1:
            merge_prompt = self.create_merge_summary_prompt()
            merged_summary = await query_executor.concurrent_query(
                [merge_prompt.format(summaries="\n".join(initial_summaries))]
            )
            current_summary = merged_summary[0]
        else:
            current_summary = initial_summaries[0]

        # 3. 如果需要，进行压缩
        if len(current_summary) > self.max_words:
            compression_prompt = self.create_compression_prompt()
            compressed_summary = await query_executor.concurrent_query(
                [compression_prompt.format(content=current_summary)]
            )
            return compressed_summary[0]

        return current_summary