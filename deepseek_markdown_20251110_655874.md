# Cryptocurrency GMMA Strategy Backtest

基于Guppy多重移动平均线(GMMA)的加密货币30分钟图表回测系统。

## 策略概述
- **交易品种**: BTC/USDT
- **时间框架**: 30分钟
- **指标**: GMMA (3,5,8,10,12,15 & 30,35,40,45,50,60)
- **回测期**: 2023年1月-2024年6月

## 策略规则
- **买入信号**: 短期组上穿长期组，且两组均线发散
- **卖出信号**: 短期组下穿长期组

## 安装依赖
```bash
pip install pandas numpy matplotlib ccxt ta-lib