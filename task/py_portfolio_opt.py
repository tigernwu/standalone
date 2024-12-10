import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
from typing import List, Dict, Any
from task.stock_price_predictor import run_stock_prediction
from core.stock.tushare_data_provider import get_technical_factor

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
    
    # 创建历史价格矩阵
    prices_matrix = pd.DataFrame()
    for symbol in assets:
        df = historical_data[symbol]
        prices_matrix[symbol] = df['close'].sort_index()
    
    # 使用最近的lookback_days数据
    prices_matrix = prices_matrix.tail(lookback_days)
    
    # 计算协方差矩阵
    S = risk_models.sample_cov(prices_matrix)
    
    # 使用预测收益作为期望收益
    mu = pd.Series(predicted_returns)
    
    # 设置优化器的通用参数
    weight_bounds = (0, 0.3)  # 单个资产权重限制在0-30%
    
    # 1. 最小方差组合
    ef_min_var = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    min_var_weights = ef_min_var.min_volatility()
    min_var_clean_weights = ef_min_var.clean_weights()
    min_var_metrics = ef_min_var.portfolio_performance(verbose=False)
    
    # 2. 最大夏普比率组合
    ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    max_sharpe_weights = ef_max_sharpe.max_sharpe()
    max_sharpe_clean_weights = ef_max_sharpe.clean_weights()
    max_sharpe_metrics = ef_max_sharpe.portfolio_performance(verbose=False)
    
    # 3. 最大收益组合（使用极高的风险容忍度来最大化收益）
    ef_max_return = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    max_return_weights = ef_max_return.max_quadratic_utility(risk_aversion=0.00001)  # 极低的风险厌恶系数
    max_return_clean_weights = ef_max_return.clean_weights()
    max_return_metrics = ef_max_return.portfolio_performance(verbose=False)
    
    # 准备结果
    portfolios_weights = {
        '最小方差组合': pd.Series(min_var_clean_weights),
        '最大夏普比率组合': pd.Series(max_sharpe_clean_weights),
        '最大收益组合': pd.Series(max_return_clean_weights)
    }
    
    # 准备汇总统计信息
    summary_stats = pd.DataFrame({
        '最小方差组合': [min_var_metrics[0], min_var_metrics[1], min_var_metrics[2]],
        '最大夏普比率组合': [max_sharpe_metrics[0], max_sharpe_metrics[1], max_sharpe_metrics[2]],
        '最大收益组合': [max_return_metrics[0], max_return_metrics[1], max_return_metrics[2]]
    }, index=['Expected Return', 'Volatility', 'Sharpe Ratio']).T
    
    return {
        'weights': portfolios_weights,
        'summary': summary_stats,
        'prices_matrix': prices_matrix,
        'expected_returns': mu,
        'covariance': S
    }

def create_optimized_portfolio(symbols: List[str], predictor_days: int = 400):
    """
    使用CatBoost预测和PyPortfolioOpt创建优化的投资组合
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
        # 可以添加可视化代码
        import matplotlib.pyplot as plt
        
        # 绘制权重分布图
        for portfolio_name, weights in results['weights'].items():
            plt.figure(figsize=(12, 6))
            weights[weights > 0.01].plot(kind='bar')
            plt.title(f'{portfolio_name}权重分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"\n绘图过程中出现错误: {str(e)}")

if __name__ == "__main__":
    runner()