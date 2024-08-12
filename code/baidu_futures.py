from datetime import datetime, timedelta
import hashlib
import random
import re
import time
from typing import List, Literal, Optional
import pandas as pd
import requests
import logging


class FuturesNewsProvider:
    def __init__(self) -> None:
        pass

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
    
    def get_futures_news(self,code: str = 'SC0', page_num: int = 0, page_size: int = 20) -> Optional[pd.DataFrame]:
        code = f"{code[:-1]}000" if code.endswith('0') else f"{code}000"
        url = 'https://finance.pae.baidu.com/vapi/getfuturesnews'

        headers = {'accept': 'application/vnd.finance-web.v1+json', 'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6', 'acs-token': '1723129376293_1723210022338_vTzYBKUWPPl0zgU6iPsEpT57CIx3qSoFqTJJbm34HW6U0BLRqJHg//N4b87dDW+7HB7kQU+lweMOObl4GhHCTqJZchF4W3rD6tddsdklL/0RLCvFrO524ZN0yVcnHwHes0aw9XiWpBbCEc59RMUwe0Loe/nRlefD0Aq35+9oywPpQQo3GG49we/PJSpWC4ZcdNFOTXhI2Ky8b2+zJCUjWqm525iQeGhdQQKQPacPzw3cR0yNTV2wQvBfgMFHAoHjMZMRTzVXxRv51oHq9BF/KQLAjCKGpQhReXxtXPp/KRsyepnFPHTxgrBQTsq0i/OSuKRx9QFFaXLGg32g+lpmi38MDHNoe/QMUfjJWCsDiVu7LmiOXOoE6IMhZjwbnHNKNw23R/m8B1mfk846AaCzwN27PHrxZKz4tEg2NA0gGJCiDdHRCKDMLgUNO0t3MO/s6s06SNCUX5WaZLPhv6kxOeUbs08PJU+aDpp1uOJQ3MkuGLTyChK9WWBKJ4xWQQlZ', 'origin': 'https://gushitong.baidu.com', 'priority': 'u=1, i', 'referer': 'https://gushitong.baidu.com/', 'sec-ch-ua': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"', 'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"Windows"', 'sec-fetch-dest': 'empty', 'sec-fetch-mode': 'cors', 'sec-fetch-site': 'same-site', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'}

        cookies = {'__bid_n': '18e6eb2425d2304866020a', 'BAIDUID': '564AD52829EF1290DDC1A20DCC14F220:FG=1', 'BAIDUID_BFESS': '564AD52829EF1290DDC1A20DCC14F220:FG=1', 'BIDUPSID': '564AD52829EF1290DDC1A20DCC14F220', 'PSTM': '1714397940', 'ZFY': '3ffAdSTQ3amiXQ393UWe0Uy1s70:BPIai4AGEBTM6yIQ:C', 'H_PS_PSSID': '60275_60287_60297_60325', 'MCITY': '-131%3A', 'BDUSS': 'X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma', 'BDUSS_BFESS': 'X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma', 'newlogin': '1', 'RT': '"z=1&dm=baidu.com&si=ec229f0d-1099-40a1-a3eb-7e95c88c7a95&ss=lzlx90cd&sl=1&tt=3p6&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=4xi"', 'ab_sr': '1.0.1_ZTJiNWQ0MTZkYWQ1NDg2OWU3MDM4MjlkOTlhMjRiMWQ3MWM5OGY5ODMxM2U5MmVhOGU4ZTg3NjJlZjI5OWVmMzgyMmQ0N2Y1NjdjYTY5Zjg2Yjg3MGE5NGVjNmI5NzA4MmJkMmM5MjBkZGI3YWQyMTViZDY0ZDBlNjZjNDcwY2ZjYTcwMWM0M2FhMTVhOGFhNDRlZmRkMzNiMzQ2MzY3MQ=='}

        params = {
            'code': code,
            'pn': page_num,
            'rn': page_size,
            'finClientType': 'pc'
        }

        try:
            response = requests.get(url, headers=headers, params=params, cookies=cookies)
            response.raise_for_status()

            data = response.json()

            if 'Result' in data and isinstance(data['Result'], list):
                df = pd.DataFrame(data['Result'])
                return df
            else:
                print("Unexpected data structure in the response")
                return None

        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

def curl_to_python_code(curl_command: str) -> str:
    # Extract URL
    url_match = re.search(r"curl '([^']+)'", curl_command)
    url = url_match.group(1) if url_match else ''

    # Extract headers
    headers = {}
    cookies = {}
    header_matches = re.findall(r"-H '([^:]+): ([^']+)'", curl_command)
    for key, value in header_matches:
        if key.lower() == 'cookie':
            cookies = {k.strip(): v.strip() for k, v in [cookie.split('=', 1) for cookie in value.split(';')]}
        else:
            headers[key] = value

    # Generate Python code
    code = f"""import requests
import pandas as pd
from typing import Optional

def get_futures_news(code: str = 'SC0', page_num: int = 0, page_size: int = 20) -> Optional[pd.DataFrame]:
    code = f"{{code[:-1]}}888" if code.endswith('0') else f"{{code}}888"
    url = 'https://finance.pae.baidu.com/vapi/getfuturesnews'
    
    headers = {headers}
    
    cookies = {cookies}
    
    params = {{
        'code': code,
        'pn': page_num,
        'rn': page_size,
        'finClientType': 'pc'
    }}
    
    try:
        response = requests.get(url, headers=headers, params=params, cookies=cookies)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Result' in data and isinstance(data['Result'], list):
            df = pd.DataFrame(data['Result'])
            return df
        else:
            print("Unexpected data structure in the response")
            return None
    
    except requests.RequestException as e:
        print(f"An error occurred: {{e}}")
        return None

# Usage example:
# df = get_futures_news('SC0')
# if df is not None:
#     print(df.head())
"""
    return code


cstr = """
curl 'https://finance.pae.baidu.com/vapi/getfuturesnews?code=RB000&pn=0&rn=20&finClientType=pc' \
  -H 'accept: application/vnd.finance-web.v1+json' \
  -H 'accept-language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6' \
  -H 'acs-token: 1723129376293_1723210022338_vTzYBKUWPPl0zgU6iPsEpT57CIx3qSoFqTJJbm34HW6U0BLRqJHg//N4b87dDW+7HB7kQU+lweMOObl4GhHCTqJZchF4W3rD6tddsdklL/0RLCvFrO524ZN0yVcnHwHes0aw9XiWpBbCEc59RMUwe0Loe/nRlefD0Aq35+9oywPpQQo3GG49we/PJSpWC4ZcdNFOTXhI2Ky8b2+zJCUjWqm525iQeGhdQQKQPacPzw3cR0yNTV2wQvBfgMFHAoHjMZMRTzVXxRv51oHq9BF/KQLAjCKGpQhReXxtXPp/KRsyepnFPHTxgrBQTsq0i/OSuKRx9QFFaXLGg32g+lpmi38MDHNoe/QMUfjJWCsDiVu7LmiOXOoE6IMhZjwbnHNKNw23R/m8B1mfk846AaCzwN27PHrxZKz4tEg2NA0gGJCiDdHRCKDMLgUNO0t3MO/s6s06SNCUX5WaZLPhv6kxOeUbs08PJU+aDpp1uOJQ3MkuGLTyChK9WWBKJ4xWQQlZ' \
  -H 'cookie: __bid_n=18e6eb2425d2304866020a; BAIDUID=564AD52829EF1290DDC1A20DCC14F220:FG=1; BAIDUID_BFESS=564AD52829EF1290DDC1A20DCC14F220:FG=1; BIDUPSID=564AD52829EF1290DDC1A20DCC14F220; PSTM=1714397940; ZFY=3ffAdSTQ3amiXQ393UWe0Uy1s70:BPIai4AGEBTM6yIQ:C; H_PS_PSSID=60275_60287_60297_60325; MCITY=-131%3A; BDUSS=X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma; BDUSS_BFESS=X56Q3pvU1ZoNFBUaVZmWHh5QjFMQWRaVzNWcXRMc0NESTJwQ25wdm9RYlVJYnRtRVFBQUFBJCQAAAAAAAAAAAEAAACgejQAd3h5MmFiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANSUk2bUlJNma; newlogin=1; RT="z=1&dm=baidu.com&si=ec229f0d-1099-40a1-a3eb-7e95c88c7a95&ss=lzlx90cd&sl=1&tt=3p6&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=4xi"; ab_sr=1.0.1_ZTJiNWQ0MTZkYWQ1NDg2OWU3MDM4MjlkOTlhMjRiMWQ3MWM5OGY5ODMxM2U5MmVhOGU4ZTg3NjJlZjI5OWVmMzgyMmQ0N2Y1NjdjYTY5Zjg2Yjg3MGE5NGVjNmI5NzA4MmJkMmM5MjBkZGI3YWQyMTViZDY0ZDBlNjZjNDcwY2ZjYTcwMWM0M2FhMTVhOGFhNDRlZmRkMzNiMzQ2MzY3MQ==' \
  -H 'origin: https://gushitong.baidu.com' \
  -H 'priority: u=1, i' \
  -H 'referer: https://gushitong.baidu.com/' \
  -H 'sec-ch-ua: "Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-site' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
"""

# code = curl_to_python_code(cstr)
# print(code)


news_loader = FuturesNewsProvider()
news_df = news_loader.get_futures_news(code='ZN', page_num=0, page_size=20)
print(news_df)