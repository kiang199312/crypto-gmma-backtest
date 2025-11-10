import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GMMAStrategy:
    """
    Guppy多重移动平均线(GMMA)策略回测类
    """
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        
    def calculate_gmma(self, df):
        """
        计算GMMA指标 - 12条EMA均线
        短期组: 3,5,8,10,12,15
        长期组: 30,35,40,45,50,60
        """
        # 短期EMA组
        short_periods = [3, 5, 8, 10, 12, 15]
        for period in short_periods:
            df[f'ema_short_{period}'] = df['close'].ewm(span=period).mean()
        
        # 长期EMA组
        long_periods = [30, 35, 40, 45, 50, 60]
        for period in long_periods:
            df[f'ema_long_{period}'] = df['close'].ewm(span=period).mean()
            
        return df
    
    def generate_signals(self, df):
        """
        生成交易信号
        """
        df['signal'] = 0
        
        for i in range(1, len(df)):
            # 计算短期组和长期组的平均值
            short_avg = np.mean([df[f'ema_short_{p}'].iloc[i] for p in [3,5,8,10,12,15]])
            long_avg = np.mean([df[f'ema_long_{p}'].iloc[i] for p in [30,35,40,45,50,60]])
            
            prev_short_avg = np.mean([df[f'ema_short_{p}'].iloc[i-1] for p in [3,5,8,10,12,15]])
            prev_long_avg = np.mean([df[f'ema_long_{p}'].iloc[i-1] for p in [30,35,40,45,50,60]])
            
            # 买入信号: 短期组上穿长期组
            if (short_avg > long_avg and 
                prev_short_avg <= prev_long_avg and
                self._is_gmma_expanding(df, i)):
                df.loc[df.index[i], 'signal'] = 1
                
            # 卖出信号: 短期组下穿长期组
            elif (short_avg < long_avg and 
                  prev_short_avg >= prev_long_avg and
                  self.position > 0):
                df.loc[df.index[i], 'signal'] = -1
                
        return df
    
    def _is_gmma_expanding(self, df, idx):
        """
        检查GMMA是否发散（短期组与长期组分离）
        """
        short_min = min([df[f'ema_short_{p}'].iloc[idx] for p in [3,5,8,10,12,15]])
        short_max = max([df[f'ema_short_{p}'].iloc[idx] for p in [3,5,8,10,12,15]])
        long_min = min([df[f'ema_long_{p}'].iloc[idx] for p in [30,35,40,45,50,60]])
        long_max = max([df[f'ema_long_{p}'].iloc[idx] for p in [30,35,40,45,50,60]])
        
        # 短期组整体在长期组上方且有明显分离
        return short_min > long_max and (short_min - long_max) > long_max * 0.002
    
    def backtest(self, df):
        """
        执行回测
        """
        print("开始GMMA策略回测...")
        
        # 计算GMMA指标
        df = self.calculate_gmma(df)
        df = self.generate_signals(df)
        
        # 初始化回测变量
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        
        entry_price = 0
        entry_idx = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # 记录权益曲线
            if self.position > 0:
                current_equity = self.capital + self.position * current_price
            else:
                current_equity = self.capital
                
            self.equity_curve.append(current_equity)
            
            # 执行交易信号
            if signal == 1 and self.position == 0:  # 买入
                self.position = self.capital * 0.95 / current_price  # 使用95%资金
                self.capital -= self.position * current_price
                entry_price = current_price
                entry_idx = i
                
                trade = {
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'position': self.position,
                    'type': 'LONG'
                }
                self.trades.append(trade)
                print(f"买入: {df.index[i]}, 价格: {current_price:.2f}, 仓位: {self.position:.4f}")
                
            elif signal == -1 and self.position > 0:  # 卖出
                pnl = (current_price - entry_price) * self.position
                return_pct = (current_price - entry_price) / entry_price * 100
                
                self.capital += self.position * current_price
                
                # 更新交易记录
                self.trades[-1].update({
                    'exit_time': df.index[i],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'hold_period': i - entry_idx
                })
                
                print(f"卖出: {df.index[i]}, 价格: {current_price:.2f}, 盈亏: {pnl:.2f} ({return_pct:.2f}%)")
                self.position = 0
        
        # 处理未平仓头寸
        if self.position > 0:
            current_price = df['close'].iloc[-1]
            pnl = (current_price - entry_price) * self.position
            self.capital += self.position * current_price
            
            self.trades[-1].update({
                'exit_time': df.index[-1],
                'exit_price': current_price,
                'pnl': pnl,
                'return_pct': (current_price - entry_price) / entry_price * 100,
                'hold_period': len(df) - 1 - entry_idx
            })
        
        return df
    
    def generate_report(self, df):
        """
        生成回测报告
        """
        if not self.trades:
            print("没有交易执行！")
            return
        
        # 计算绩效指标
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100
        winning_trades = [t for t in self.trades if 'pnl' in t and t['pnl'] > 0]
        losing_trades = [t for t in self.trades if 'pnl' in t and t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')
        
        max_drawdown = self.calculate_max_drawdown()
        
        print("\n" + "="*50)
        print("GMMA策略回测报告")
        print("="*50)
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"最终资金: ${self.equity_curve[-1]:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"总交易次数: {len(self.trades)}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"平均盈利: ${avg_win:.2f}")
        print(f"平均亏损: ${avg_loss:.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print("="*50)
    
    def calculate_max_drawdown(self):
        """计算最大回撤"""
        peak = self.equity_curve[0]
        max_dd = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def plot_results(self, df):
        """
        绘制回测结果图表
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 价格和GMMA图表
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1, color='black')
        
        # 绘制短期GMMA组
        short_periods = [3, 5, 8, 10, 12, 15]
        for period in short_periods:
            ax1.plot(df.index, df[f'ema_short_{period}'], 
                    alpha=0.6, linewidth=0.8)
        
        # 绘制长期GMMA组
        long_periods = [30, 35, 40, 45, 50, 60]
        for period in long_periods:
            ax1.plot(df.index, df[f'ema_long_{period}'], 
                    alpha=0.6, linewidth=0.8)
        
        # 标记交易信号
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   color='green', marker='^', s=100, label='Buy', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   color='red', marker='v', s=100, label='Sell', zorder=5)
        
        ax1.set_title('GMMA Strategy - Price and Signals')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 权益曲线
        ax2.plot(df.index, self.equity_curve, label='Equity Curve', linewidth=2, color='blue')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Equity (USDT)')
        ax2.grid(True, alpha=0.3)
        
        # 回撤图表
        drawdown = self.calculate_drawdown_series()
        ax3.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax3.plot(df.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gmma_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_drawdown_series(self):
        """计算回撤序列"""
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak * 100
        return drawdown

def generate_sample_data():
    """
    生成样本数据（在实际应用中应从API或CSV文件获取）
    """
    print("生成样本数据...")
    
    # 创建日期范围（30分钟间隔）
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='30T')
    
    # 生成模拟价格数据（随机游走）
    np.random.seed(42)
    price = 25000  # 起始价格
    prices = []
    
    for _ in range(len(dates)):
        # 随机波动
        change = np.random.normal(0, 0.002)  # 平均波动0.2%
        price *= (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(prices))
    }, index=dates)
    
    return df

def main():
    """
    主函数
    """
    print("加密货币GMMA策略回测系统")
    print("=" * 40)
    
    # 生成或加载数据
    df = generate_sample_data()  # 在实际使用中替换为真实数据
    
    # 初始化策略
    strategy = GMMAStrategy(initial_capital=10000)
    
    # 执行回测
    df = strategy.backtest(df)
    
    # 生成报告
    strategy.generate_report(df)
    
    # 绘制图表
    strategy.plot_results(df)
    
    # 保存结果到CSV
    df.to_csv('backtest_results.csv')
    print("\n回测结果已保存到 backtest_results.csv")

if __name__ == "__main__":
    main()