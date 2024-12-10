import numpy as np
import pandas as pd
import riskfolio as rp
import seaborn as sns
from typing import List, Dict, Any
from task.stock_price_predictor import run_stock_prediction
from core.stock.tushare_data_provider import get_technical_factor
from matplotlib import pyplot as plt

def plot_correlation_matrix(returns_data: pd.DataFrame):
    """绘制相关性热图"""
    corr = returns_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f', 
                square=True,
                cbar_kws={"shrink": .5})
    plt.title('资产收益率相关性热图')
    plt.tight_layout()
    return plt.gca()

def optimize_portfolio(stock_predictions: Dict[str, Dict], 
                      historical_data: Dict[str, pd.DataFrame],
                      lookback_days: int = 252) -> Dict[str, Any]:
    """
    使用预测收益和历史数据优化投资组合配置
    
    参数:
        stock_predictions: CatBoost模型预测结果字典
        historical_data: 每只股票的历史价格数据字典
        lookback_days: 用于计算协方差的历史数据天数
    """
    
    # 提取预测收益和准备数据
    assets = []
    predicted_returns = {}
    
    for symbol, pred in stock_predictions.items():
        if 'error' not in pred:
            assets.append(symbol)
            predicted_returns[symbol] = pred['price_change'] / 100
    
    # 创建历史价格矩阵和收益率矩阵
    prices_matrix = pd.DataFrame()
    for symbol in assets:
        df = historical_data[symbol]
        prices_matrix[symbol] = df['close'].sort_index()
    
    # 使用最近的lookback_days数据
    prices_matrix = prices_matrix.tail(lookback_days)
    
    # 计算历史收益率
    returns_matrix = prices_matrix.pct_change(fill_method=None).dropna()
    
    # 创建预期收益率Series并确保索引匹配
    mu = pd.Series(predicted_returns, index=returns_matrix.columns)
    
    # 创建Portfolio对象
    port = rp.Portfolio(returns=returns_matrix)
    
    # 更新预期收益为预测值
    port.mu = mu
    
    # 计算历史协方差矩阵
    port.cov = returns_matrix.cov()
    
    print("\n开始优化...")
    # 1. 最小方差组合
    w_min_var = port.optimization(model='Classic', rm='MV', obj='MinRisk')
    print("最小方差组合优化成功")
    
    # 2. 最大夏普比率组合
    w_max_sharpe = port.optimization(model='Classic', rm='MV', obj='Sharpe')
    print("最大夏普比率组合优化成功")
    
    # 3. 最大效用组合
    w_max_utility = port.optimization(model='Classic', rm='MV', obj='Utility')
    print("最大效用组合优化成功")
    
    # 准备结果
    portfolios_weights = {
        '最小方差组合': pd.Series(w_min_var.iloc[:, 0], index=returns_matrix.columns),
        '最大夏普比率组合': pd.Series(w_max_sharpe.iloc[:, 0], index=returns_matrix.columns),
        '最大效用组合': pd.Series(w_max_utility.iloc[:, 0], index=returns_matrix.columns)
    }
    
    # 计算各组合的表现指标
    stats_list = []
    for name, weights in portfolios_weights.items():
        # 计算预期收益
        expected_return = (weights * mu).sum()
        # 计算波动率
        volatility = np.sqrt(weights @ returns_matrix.cov() @ weights) * np.sqrt(252)
        # 计算夏普比率
        sharpe_ratio = (expected_return - 0.03) / volatility
        
        stats_list.append({
            'Expected Return': expected_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio
        })
    
    # 创建汇总统计表
    summary_stats = pd.DataFrame(stats_list, 
                               index=['最小方差组合', '最大夏普比率组合', '最大效用组合'])
    
    return {
        'weights': portfolios_weights,
        'summary': summary_stats,
        'portfolio': port,
        'returns_data': returns_matrix
    }

def create_optimized_portfolio(symbols: List[str], predictor_days: int = 400):
    """
    使用CatBoost预测和Riskfolio创建优化的投资组合
    """
    # 获取所有股票的预测结果
    stock_predictions = run_stock_prediction(
        symbols,
        days=predictor_days,
        optimize=False,
        max_workers=len(symbols)
    )
    
    # 获取所有股票的历史数据
    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = get_technical_factor(symbol, days=predictor_days)
    
    # 优化投资组合
    results = optimize_portfolio(stock_predictions, historical_data)
    
    return results

# 定义股票列表
symbols = [
    "300750", "603993", "600519", "002241", "601318", 
    "600030", "601919", "002119", "605117", "601127",
    "300641", "603683", "600941", "600004", "600032",
    "600787", "300502", "600182", "002008", "002594",
    "600066", "603496", "002352", "003015", "001696"
]

def runner():
    """投资组合优化的运行函数"""
    results = create_optimized_portfolio(symbols)
    
    print("\n=== 投资组合优化结果 ===")
    
    # 打印每个组合的权重分配
    for portfolio_name, weights in results['weights'].items():
        print(f"\n{portfolio_name}配置:")
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
        for symbol, weight in sorted_weights:
            if weight > 0.01:  # 只显示权重大于1%的持仓
                print(f"{symbol}: {weight*100:.1f}%")
    
    # 打印汇总统计信息
    print("\n=== 组合表现指标 ===")
    summary = results['summary']
    
    # 打印每个投资组合的关键指标
    print("\n投资组合指标比较:")
    for portfolio in summary.index:
        print(f"\n{portfolio}:")
        print(f"预期年化收益率: {summary.loc[portfolio, 'Expected Return']*100:.2f}%")
        print(f"年化波动率: {summary.loc[portfolio, 'Volatility']*100:.2f}%")
        print(f"夏普比率: {summary.loc[portfolio, 'Sharpe Ratio']:.2f}")

    try:
        # 添加可视化代码
        import matplotlib.pyplot as plt
        
        # 1. 绘制权重分布图
        for portfolio_name, weights in results['weights'].items():
            plt.figure(figsize=(12, 6))
            weights[weights > 0.01].plot(kind='bar')
            plt.title(f'{portfolio_name}权重分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # 2. 绘制相关性热图
        plot_correlation_matrix(results['returns_data'])
        plt.show()
            
    except Exception as e:
        print(f"\n绘图过程中出现错误: {str(e)}")

if __name__ == "__main__":
    runner()