import numpy as np
import pandas as pd
from skfolio import Population, Portfolio, RiskMeasure, PerfMeasure, RatioMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
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
    
    # 提取预测收益
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
    
    # 计算历史收益率
    returns_df = prices_to_returns(prices_matrix.tail(lookback_days))
    
    # 1. 最小方差组合
    min_var_model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        min_weights=0,
        max_weights=0.3
    )
    min_var_model.fit(returns_df)
    min_variance_portfolio = Portfolio(
        returns_df,
        weights=min_var_model.weights_,
        name="最小方差组合"
    )
    
    # 2. 最大收益组合
    max_return_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        min_weights=0,
        max_weights=0.3
    )
    max_return_model.fit(returns_df)
    max_return_portfolio = Portfolio(
        returns_df,
        weights=max_return_model.weights_,
        name="最大收益组合"
    )
    
    # 3. 最大夏普比率组合
    max_sharpe_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
        min_weights=0,
        max_weights=0.3
    )
    max_sharpe_model.fit(returns_df)
    max_sharpe_portfolio = Portfolio(
        returns_df,
        weights=max_sharpe_model.weights_,
        name="最大夏普比率组合"
    )
    
    # 创建投资组合集合进行比较
    population = Population([min_variance_portfolio, max_return_portfolio, max_sharpe_portfolio])
    
    # 准备结果报告
    portfolios_weights = {
        '最小方差组合': pd.Series(min_variance_portfolio.weights, index=returns_df.columns),
        '最大收益组合': pd.Series(max_return_portfolio.weights, index=returns_df.columns),
        '最大夏普比率组合': pd.Series(max_sharpe_portfolio.weights, index=returns_df.columns)
    }
    
    # 获取投资组合汇总信息
    summary_stats = population.summary()
    
    return {
        'weights': portfolios_weights,
        'summary': summary_stats,
        'population': population,
        'returns_data': returns_df  # 保存收益率数据供后续使用
    }

def create_optimized_portfolio(symbols: List[str], predictor_days: int = 400):
    """
    使用CatBoost预测和skfolio创建优化的投资组合
    """
    # 获取所有股票的预测结果
    stock_predictions = run_stock_prediction(
        symbols,
        days=predictor_days,
        optimize=False, #改成True，优化参数
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
    
    # 对每个投资组合打印关键指标
    metrics = ['Mean', 'Variance', 'Sharpe Ratio', 'Max Drawdown']
    print("\n关键指标比较:")
    for metric in metrics:
        if metric in summary.columns:
            print(f"\n{metric}:")
            for portfolio in summary.index:
                print(f"{portfolio}: {summary.loc[portfolio, metric]:.4f}")

    try:
        # 尝试绘制投资组合构成
        results['population'].plot_composition()
        print("\n投资组合构成图已生成")
        
        # 尝试绘制累计收益率
        results['population'].plot_cumulative_returns()
        print("累计收益率图已生成")
    except Exception as e:
        print(f"\n绘图过程中出现错误: {str(e)}")

if __name__ == "__main__":
    runner()