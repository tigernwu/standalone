from datetime import datetime
import hashlib
import json
import os
import random
import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote
import requests
from tenacity import retry, stop_after_attempt, wait_fixed



class BaiduFinanceAPI:
    def __init__(self):
        self.base_urls = {
            'news': 'https://finance.pae.baidu.com/selfselect/news',
            'analysis': 'https://finance.pae.baidu.com/vapi/v1/analysis',
            'express_news': 'https://finance.pae.baidu.com/selfselect/expressnews',
            'finance_calendar': 'https://finance.pae.baidu.com/api/financecalendar',
            'hotrank': 'https://finance.pae.baidu.com/vapi/v1/hotrank',
            'recommendation_list': 'https://finance.pae.baidu.com/selfselect/listsugrecomm',
            'sentiment_rank': 'https://finance.pae.baidu.com/vapi/sentimentrank',
            'analysis_rank': 'https://finance.pae.baidu.com/vapi/v1/analysisrank',
            'report_basics': 'https://finance.pae.baidu.com/vapi/v1/getreportbasics',
            'stock_recommend':'https://finance.pae.baidu.com/vapi/v1/stock/recommend'
        }
        self._add_header()

    def _add_header(self):
        header_file = './json/baidu_headers.json'
        if os.path.exists(header_file):
            with open(header_file, 'r') as f:
                self.headers = json.load(f)
        else:
            self.headers = {
                'accept': 'application/vnd.finance-web.v1+json',
                'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                'origin': 'https://gushitong.baidu.com',
                'referer': 'https://gushitong.baidu.com/',
                'sec-ch-ua': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-site',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
            }

    def save_headers_from_command(self, command_string: str):
        """
        从 cURL 或 PowerShell 命令字符串中提取 headers 信息并保存到文件。

        :param command_string: cURL 或 PowerShell 命令字符串
        """
        headers = {}

        # 提取 headers
        if 'curl' in command_string:
            # 处理 cURL 命令
            header_pattern = r'-H ["\']([^:]+):\s*([^"\']+)["\']'
        else:
            # 处理 PowerShell 命令
            header_pattern = r'"([^"]+)"="([^"]+)"'

        matches = re.findall(header_pattern, command_string)
        for key, value in matches:
            # 移除可能的引号
            key = key.strip("'\"")
            value = value.strip("'\"")

            # 过滤掉值为空字符串的 header
            if value: 
                headers[key] = value

        # 特殊处理 cookie
        cookie_match = re.search(r'-H ["\']cookie:\s*([^"\']+)["\']', command_string)
        if cookie_match:
            cookie_value = cookie_match.group(1).strip("'\"")
            if cookie_value:  # 只有当 cookie 值不为空时才添加
                headers['cookie'] = cookie_value

        # 保存到文件
        os.makedirs('./json', exist_ok=True)
        with open('./json/baidu_headers.json', 'w', encoding='utf-8') as f:
            json.dump(headers, f, ensure_ascii=False, indent=2)

        print("Headers have been saved to './json/baidu_headers.json'")

    def generate_acs_token(self):
        current_time = int(time.time() * 1000)
        random_num = random.randint(1000000000000000, 9999999999999999)  # 16位随机数

        part1 = str(current_time)
        part2 = str(random_num)
        part3 = "1"

        token = f"{part1}_{part2}_{part3}"

        md5 = hashlib.md5()
        md5.update(token.encode('utf-8'))
        hash_value = md5.hexdigest()

        # 添加额外的随机字符串来增加长度
        extra_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=20))

        final_token = f"{token}_{hash_value}_{extra_chars}"

        return final_token

    def get_stock_recommendations(self, code: str, market: str = 'ab', sort_key: str = 'market_value', 
                                  sort_type: str = 'down', page_num: int = 0, page_size: int = 10) -> Optional[List[Dict]]:
        """
        获取股票推荐列表。

        :param code: 股票代码
        :param market: 市场类型，默认为 'ab'
        :param sort_key: 排序关键字，默认为 'market_value'
        :param sort_type: 排序类型，默认为 'down'
        :param page_num: 页码，默认为 0
        :param page_size: 每页数量，默认为 6
        :return: 推荐股票列表，如果出错则返回 None
        """
        url = self.base_urls['stock_recommend']
        params = {
            'code': code,
            'market': market,
            'sortKey': sort_key,
            'sortType': sort_type,
            'pn': page_num,
            'rn': page_size,
            'finClientType': 'pc'
        }
        headers = self.headers.copy()
        headers['acs-token'] = self.generate_acs_token()

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data['ResultCode'] == 0 and 'Result' in data and 'list' in data['Result']:
                return data['Result']['list']
            else:
                print(f"Error in API response: {data.get('ResultCode', 'Unknown error')}")
                return None
        except requests.RequestException as e:
            print(f"Error fetching stock recommendations: {e}")
            return None

    def get_report_basics(self, code: str, name: str):
        """
        获取股票的报告基础信息。

        :param code: 股票代码
        :param name: 股票名称
        :return: 处理后的报告基础信息
        """
        url = self.base_urls['report_basics']
        params = {
            'code': code,
            'name': name,
            'finClientType': 'pc'
        }
        headers = self.headers.copy()
        headers['acs-token'] = self.generate_acs_token()

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data['ResultCode'] == 0 and 'Result' in data:
                result = data['Result']
                return {
                    'description': result.get('desc', ''),
                    'pdf_url': result.get('pdfUrl', '')
                }
            else:
                print(f"Error in API response: {data.get('ResultCode', 'Unknown error')}")
                return None
        except requests.RequestException as e:
            print(f"Error fetching report basics: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_news(self, rn=6, pn=0):
        """
        获取新闻信息。

        Args:
            rn (int): 每页显示的新闻数量，默认值为6。
            pn (int): 页码，默认值为0。

        Returns:
            list: 包含新闻信息的列表，每个元素是一个字典，包含标题、内容、发布时间、标签和提供者。
        """
        params = {
            'rn': rn,
            'pn': pn,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['news'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_news(response.json())
        else:
            response.raise_for_status()

    def parse_news(self, data):
        news_list = []
        if "Result" in data and "tabs" in data["Result"]:
            for tab in data["Result"]["tabs"]:
                if "contents" in tab and "list" in tab["contents"]:
                    for item in tab["contents"]["list"]:
                        content_data = " ".join([c['data'] for c in item.get("content", {}).get("items", []) if c.get('data')])
                        publish_time = datetime.fromtimestamp(int(item.get("publish_time", 0)))
                        news_item = {
                            "title": item.get("title", ""),
                            "content": content_data,
                            "ptime": publish_time,
                            "tag": item.get("tag", ""),
                            "provider": item.get("provider", "")
                        }
                        news_list.append(news_item)
        return news_list

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_analysis(self, code='000725', market='ab'):
        """
        获取指定股票的分析数据。

        Args:
            code (str): 股票代码，默认为 '000725'。
            market (str): 市场类型（ab、hk、us等），默认为 'ab'。

        Returns:
            str: 解析后的分析信息，以可读文本形式返回。
        """
        params = {
            'code': code,
            'market': market,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['analysis'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_analysis(response.json())
        else:
            response.raise_for_status()

    def parse_analysis(self, data):
        result = data.get("Result", {})
        if result is None or result == {}:
            return "数据不可用"
        output = []

        synthesis_score = result.get("synthesisScore", {})
        technology_score = result.get("technologyScore", {})
        capital_score = result.get("capitalScore", {})
        market_score = result.get("marketScore", {})
        finance_score = result.get("financeScore", {})

        if synthesis_score:
            output.append(f"综合得分: {synthesis_score.get('rating', 'N/A')} ({synthesis_score.get('desc', 'N/A')})")
            output.append(f"行业排名: {synthesis_score.get('industryRanking', 'N/A')} / {synthesis_score.get('firstIndustryName', 'N/A')}")
            output.append(f"更新时间: {synthesis_score.get('updateTime', 'N/A')}")
            output.append("")
        else:
            output.append("综合得分: 数据不可用")
            output.append("")

        if technology_score:
            output.append(f"技术面: {technology_score.get('score', 'N/A')} ({technology_score.get('desc', 'N/A')})")
            increase_items = technology_score.get('increase', {}).get('items', [])
            if increase_items:
                output.append(f"近5日累计涨跌幅: {', '.join([item.get('increase', 'N/A') for item in increase_items])}")
            output.append("")
        else:
            output.append("技术面: 数据不可用")
            output.append("")

        if capital_score:
            output.append(f"资金面: {capital_score.get('score', 'N/A')} ({capital_score.get('desc', 'N/A')})")
            fundflow = capital_score.get('fundflow', {}).get('body', [])
            for flow in fundflow:
                output.append(f"{flow.get('name', 'N/A')}: 净流入 {flow.get('in', 'N/A')}, 净占比 {flow.get('out', 'N/A')}")
            output.append("")

        if market_score:
            output.append(f"市场面: {market_score.get('score', 'N/A')} ({market_score.get('desc', 'N/A')})")
            output.append("")
        if finance_score:
            output.append(f"财务面: {finance_score.get('score', 'N/A')} ({finance_score.get('desc', 'N/A')})")
            rating_content = finance_score.get('ratingContent', {}).get('list', [])
            for section in rating_content:
                output.append(f"{section.get('title', 'N/A')}:")
                for body in section.get('body', []):
                    output.append(f"  {body.get('name', 'N/A')}: 本期 {body.get('thisIssue', 'N/A')}, 上期 {body.get('previousPeriod', 'N/A')}, 行业排名 {body.get('industryRanking', 'N/A')}")
            output.append("")

        return "\n".join(output)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_express_news(self, rn=10, pn=0, finance_type='stock', code='000725'):
        """
        获取指定股票的快讯新闻。

        Args:
            rn (int): 每页显示的新闻数量，默认值为10。
            pn (int): 页码，默认值为0。
            finance_type (str): 财务类型（例如 'stock'）。
            code (str): 股票代码，默认为 '000725'。

        Returns:
            list: 包含快讯新闻的列表，每个元素是一个字典，包含标题、内容、发布时间、标签和提供者。
        """
        params = {
            'rn': rn,
            'pn': pn,
            'financeType': finance_type,
            'code': code,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['express_news'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_express_news(response.json())
        else:
            response.raise_for_status()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_express_news_v2(self, rn=6, pn=0, tag='A股'):
        """
        获取按标签筛选的快讯新闻。

        Args:
            rn (int): 每页显示的新闻数量，默认值为6。
            pn (int): 页码，默认值为0。
            tag (str): 标签筛选，默认为 'A股'。

        Returns:
            list: 包含快讯新闻的列表，每个元素是一个字典。
        """
        params = {
            'rn': rn,
            'pn': pn,
            'tag': quote(tag),  # URL encode the tag
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['express_news'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_express_news(response.json())
        else:
            response.raise_for_status()

    def parse_express_news(self, data):
        news_list = []
        if "Result" in data and "content" in data["Result"] and "list" in data["Result"]["content"]:
            for item in data["Result"]["content"]["list"]:
                news_item = {
                    "title": item.get("title", ""),
                    "content": " ".join([c['data'] for c in item.get("content", {}).get("items", []) if c.get('data')]),
                    "ptime": datetime.fromtimestamp(int(item.get("publish_time", 0))),
                    "tag": item.get("tag", ""),
                    "provider": item.get("provider", "")
                }
                news_list.append(news_item)
        return news_list

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_finance_calendar(self, start_date, end_date, cate='economic_data'):
        """
        获取指定日期范围内的财经日历数据。

        Args:
            start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
            end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
            cate (str): 财经日历类型，默认为 'economic_data'。

        Returns:
            list: 包含财经日历事件的列表，每个元素是一个字典。
        """
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'market': '',
            'cate': cate,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['finance_calendar'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_finance_calendar(response.json())
        else:
            response.raise_for_status()

    def parse_finance_calendar(self, data):
        calendar_list = []
        if "Result" in data and "data" in data["Result"]:
            for event in data["Result"]["data"].get("list", []):
                event_item = {
                    "date": event.get("date", ""),
                    "time": event.get("time", ""),
                    "countryIcon": event.get("countryIcon", ""),
                    "title": event.get("title", ""),
                    "star": event.get("star", ""),
                    "formerVal": event.get("formerVal", ""),
                    "indicateVal": event.get("indicateVal", ""),
                    "pubVal": event.get("pubVal", ""),
                    "positive": event.get("positive", ""),
                    "negative": event.get("negative", ""),
                    "region": event.get("region", ""),
                    "weight": event.get("weight", "")
                }
                calendar_list.append(event_item)
        return calendar_list

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))  
    def fetch_hotrank(self, day, hour=7, pn=0, rn=100, market='ab', type='hour'):
        """
        获取指定日期和时间的热门股票排名。

        Args:
            day (str): 日期，格式为 'YYYYMMDD'。
            hour (str): 小时，格式为 'HH'。
            pn (int): 页码，默认值为0。
            rn (int): 每页显示的股票数量，默认值为100。
            market (str): 市场类型，默认为 'ab'。
            type (str): 排行榜类型，默认为 'hour'。

        Returns:
            list: 包含热门股票排名的列表，每个元素是一个字典。
        """
        params = {
            'tn': 'wisexmlnew',
            'dsp': 'iphone',
            'product': 'stock',
            'day': day,
            'hour': hour,
            'pn': pn,
            'rn': rn,
            'market': market,
            'type': type,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['hotrank'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_hotrank(response.json())
        else:
            response.raise_for_status()

    def parse_hotrank(self, data):
        hotrank_list = []
        if "Result" in data and "body" in data["Result"]:
            for item in data["Result"]["body"]:
                rank_item = {
                    "name": item[0],       # 股票名字
                    "change": item[1],     # 涨跌幅
                    "sector": item[2],     # 板块
                    "code": item[3],       # 代码
                    "price": item[4],      # 价格
                    "market": item[5],     # 市场
                    "rank_change": item[6],# 名次变化
                    "region": item[7],     # 地区
                    "heat": item[8]        # 热度
                }
                hotrank_list.append(rank_item)
        return hotrank_list

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_recommendation_list(self, day, hour=7, pn=0, rn=100, market='ab', type='hour'):
        """
        获取指定日期和时间的股票推荐列表。

        Args:
            day (str): 日期，格式为 'YYYYMMDD'。
            hour (str): 小时，格式为 'HH'。
            pn (int): 页码，默认值为0。
            rn (int): 每页显示的股票数量，默认值为100。
            market (str): 市场类型，默认为 'ab'。
            type (str): 推荐类型，默认为 'hour'。

        Returns:
            list: 包含股票推荐的列表，每个元素是一个字典。
        """
        params = {
            'tn': 'wisexmlnew',
            'dsp': 'iphone',
            'product': 'search',
            'day': day,
            'hour': hour,
            'pn': pn,
            'rn': rn,
            'market': market,
            'type': type,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['recommendation_list'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_recommendation_list(response.json())
        else:
            response.raise_for_status()

    def parse_recommendation_list(self, data):
        recommendation_list = []
        if "Result" in data and "body" in data["Result"]:
            for item in data["Result"]["body"]:
                recommendation_item = {
                    "name": item[0],          # 股票名字
                    "change": item[1],        # 涨跌幅
                    "heat": item[2],          # 综合热度
                    "sector_name": item[3],   # 所属板块名称
                    "code": item[4],          # 市场代码
                    "sector_code": item[5],   # 所属板块代码
                    "rank_change": item[6],   # 排名变化
                    "continued_ranking": item[7] # 是否连续上榜
                }
                recommendation_list.append(recommendation_item)
        return recommendation_list

    # 添加 fetch_sentiment_rank 方法
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_sentiment_rank(self, pn=0, rn=10, market='ab', finance_type='stock'):
        """
        获取市场情绪排名数据。

        Args:
            pn (int): 页码，默认值为0。
            rn (int): 每页显示的排名数量，默认值为10。
            market (str): 市场类型。
            finance_type (str): 财务类型。

        Returns:
            list: 包含市场情绪排名的列表，每个元素是一个字典。
        """
        params = {
            'pn': pn,
            'rn': rn,
            'market': market,
            'financeType': finance_type,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['sentiment_rank'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_sentiment_rank(response.json())
        else:
            response.raise_for_status()

    # 添加 parse_sentiment_rank 方法
    def parse_sentiment_rank(self, data):
        sentiment_rank_list = []
        if "Result" in data:
            for item in data["Result"]:
                if "aiSentimentRankInfo" in item and "body" in item["aiSentimentRankInfo"]:
                    for sentiment in item["aiSentimentRankInfo"]["body"]:
                        sentiment_item = {
                            "name": sentiment.get("name", ""),
                            "code": sentiment.get("code", ""),
                            "exchange": sentiment.get("exchange", ""),
                            "market": sentiment.get("market", ""),
                            "plate": sentiment.get("plate", ""),
                            "plateCode": sentiment.get("plateCode", ""),
                            "rankDiff": sentiment.get("rankDiff", 0),
                            "ratio": sentiment.get("ratio", ""),
                            "heat": sentiment.get("heat", 0),
                            "goodNewsPercent": sentiment.get("goodNewsPercent", ""),
                            "middleNewsPercent": sentiment.get("middleNewsPercent", ""),
                            "badNewsPercent": sentiment.get("badNewsPercent", "")
                        }
                        sentiment_rank_list.append(sentiment_item)
        return sentiment_rank_list

    # 添加 fetch_analysis_rank 方法
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def fetch_analysis_rank(self, pn=0, rn=30, market='ab', first_industry='', second_industry='', order_by='', order=''):
        """
        获取分析排行榜数据。

        Args:
            pn (int): 页码，默认值为0。
            rn (int): 每页显示的排名数量，默认值为30。
            market (str): 市场类型，默认为 'ab'。
            first_industry (str): 一级行业筛选，默认为空。
            second_industry (str): 二级行业筛选，默认为空。
            order_by (str): 排序字段，默认为空。
            order (str): 排序顺序，默认为空。

        Returns:
            list: 包含分析排行榜的列表，每个元素是一个字典。
        """
        params = {
            'product': 'analysis',
            'market': market,
            'firstIndustry': first_industry,
            'secondIndustry': second_industry,
            'orderBy': order_by,
            'order': order,
            'pn': pn,
            'rn': rn,
            'finClientType': 'pc'
        }
        self.headers['acs-token'] = self.generate_acs_token()

        response = requests.get(self.base_urls['analysis_rank'], headers=self.headers, params=params)
        if response.status_code == 200:
            return self.parse_analysis_rank(response.json())
        else:
            response.raise_for_status()

    # 添加 parse_analysis_rank 方法
    def parse_analysis_rank(self, data):
        analysis_rank_list = []
        if "Result" in data and "body" in data["Result"]:
            for item in data["Result"]["body"]:
                analysis_item = {
                    "code": item[0],
                    "name": item[1],
                    "market": item[2],
                    "rank_change": item[3],
                    "synthesis_score": item[4],
                    "technology_score": item[5],
                    "capital_score": item[6],
                    "market_score": item[7],
                    "finance_score": item[8],
                    "sector": item[9],
                    "sector_code": item[10],
                    "market_type": item[11]
                }
                analysis_rank_list.append(analysis_item)
        return analysis_rank_list