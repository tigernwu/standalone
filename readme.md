# 扣德探索的代码库

## 目录
**task目录**
* "stock_price_predictor"  catboost 模型股票价格预测+参数优化
* "cls_news"                LLM新闻分析
* "finance_analysis"        LLM股票分析
* "random_forest"           随机森林预测上证指数
* "prophet_predict"         Prophet模型预测上证指数
* "arima_predict"           ARIMA模型预测上证指数
* "gbm_predict"             GBM模型预测上证指数
* "light_gbm_predict"       LightGBM模型预测上证指数
* "xgboost_predict"         XGBoost模型预测上证指数
* "gbdt_predict"            GBDT模型预测上证指数
* "cat_boost_predict"       CatBoost模型预测上证指数
* "llm_index_predict"       LLM预测上证指数
* "svm_predict"             SVM模型预测上证指数
* "decisiontree_predict"    决策树模型预测上证指数
    
**code目录**
* baidu_futures             百度财经期货数据   
* cls_news                  LLM新闻分析
* corr_fin_futures          股指期货和国债相关度
* llm_predict               LLM预测
* report_rc                 投研解读
* technical_analysis        技术分析    

## 下载
```bash
git clone https://github.com/Dong-Zhang/stock_analysis.git
```

## 配置
需要注册一个LLM的APIkey，推荐deepseek，minimax因为它的价格比较便宜，但是效果比较好。

```text
[Default]
llm_api = MiniMaxClient
llm_cheap_api = CheapClaude
embedding_api = BGELargeZhAPI
ranker_api = BaiduBCEReranker
talker = CliTalker
project_id = 
ts_key = 
aws_access_key_id =             #aws的密钥
aws_secret_access_key =         #aws的access_key
AZURE_OPENAI_API_KEY =          #azure的密钥
AZURE_OPENAI_ENDPOINT =         #azure的endpoint
GAODE_MAP_API_KEY =             #高德地图的密钥
OPEN_WEATHER_API_KEY =          #openweather的密钥
ERNIE_API_KEY =                 #文心一言的密钥
ERNIE_SERCRET_KEY =             #文心一言的secret_key
glm_api_key =                   #glm的密钥
deep_seek_api_key =             #deepseek的密钥
moonshot_api_key =              #moonshot的密钥
DASHSCOPE_API_KEY =             #阿里云百炼的密钥
baichuan_api_key =              #百川的密钥
volcengine_api_key =            #火山云的密钥
volcengine_embedding =          #火山云的embedding
volcengine_doubao =             #火山云的doubao接入点
volcengine_doubao_32k =         #火山云的doubao接入点
minimax_api_key =               #minimax的密钥
zero_one_api_key =              #zero one的密钥
OPENAI_API_KEY =                #openai的密钥
hunyuan_SecretId =              #腾讯混元的id
hunyuan_SecretKey =             #腾讯混元的key
xunfei_spark_api_key =          #讯飞的api_key  
xunfei_spark_secret_key =       #讯飞的secret_key
```     

## 安装
```shell
install.bat
```

## 运行
```shell
python task.py task_name
```
code目录下的代码需要单独运行
修改run.py带代码，然后运行
```shell
python run.py
```
