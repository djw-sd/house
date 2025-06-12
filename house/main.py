# 广东省房价影响因素分析项目
# Author: 邓健文
# Date: 2025-06-12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class GuangdongHousingAnalysis:
    def __init__(self):
        self.data = None
        self.cities = ['广州', '深圳', '珠海', '汕头', '佛山', '韶关', '湛江', '肇庆', 
                      '江门', '茂名', '惠州', '梅州', '汕尾', '河源', '阳江', '清远',
                      '东莞', '中山', '潮州', '揭阳', '云浮']
        
        # 创建保存图片的文件夹
        self.output_dir = 'housing_analysis_plots'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def generate_mock_data(self, n_samples=2000):
        """
        生成模拟的广东省房价数据
        """
        np.random.seed(42)
        
        # 基础数据
        cities = np.random.choice(self.cities, n_samples)
        
        # 为不同城市设置基础房价水平
        city_base_price = {
            '深圳': 65000, '广州': 45000, '珠海': 30000, '东莞': 25000, '中山': 20000,
            '佛山': 22000, '惠州': 18000, '江门': 15000, '肇庆': 12000, '汕头': 10000,
            '韶关': 8000, '湛江': 9000, '茂名': 8500, '梅州': 7000, '汕尾': 8000,
            '河源': 7500, '阳江': 8000, '清远': 9000, '潮州': 7500, '揭阳': 7000, '云浮': 6500
        }
        
        data = []
        
        for i in range(n_samples):
            city = cities[i]
            base_price = city_base_price[city]
            
            # 房屋特征
            area = np.random.normal(95, 30)  # 面积
            area = max(40, min(300, area))  # 限制范围
            
            age = np.random.exponential(8)  # 房龄
            age = max(0, min(30, age))
            
            floor = np.random.randint(1, 31)  # 楼层
            total_floors = max(floor, np.random.randint(floor, 35))  # 总楼层
            
            # 区域特征
            cbd_distance = np.random.exponential(15)  # 距离CBD距离(公里)
            subway_distance = np.random.exponential(2)  # 距离地铁距离(公里)
            school_count = np.random.poisson(3)  # 周边学校数量
            hospital_count = np.random.poisson(2)  # 周边医院数量
            
            # 经济指标
            gdp_per_capita = np.random.normal(80000, 20000)  # 人均GDP
            gdp_per_capita = max(30000, gdp_per_capita)
            
            population_density = np.random.normal(800, 300)  # 人口密度(人/平方公里)
            population_density = max(100, population_density)
            
            # 房屋类型
            property_type = np.random.choice(['住宅', '公寓', '别墅'], p=[0.7, 0.25, 0.05])
            decoration = np.random.choice(['毛坯', '简装', '精装', '豪装'], p=[0.2, 0.3, 0.4, 0.1])
            
            # 房价计算 (考虑多种因素)
            price_per_sqm = base_price
            
            # 面积影响
            if area > 120:
                price_per_sqm *= 1.1
            elif area < 70:
                price_per_sqm *= 0.95
            
            # 房龄影响
            price_per_sqm *= (1 - age * 0.015)
            
            # 位置影响
            price_per_sqm *= (1 - cbd_distance * 0.02)
            price_per_sqm *= (1 - subway_distance * 0.05)
            
            # 周边设施影响
            price_per_sqm *= (1 + school_count * 0.02)
            price_per_sqm *= (1 + hospital_count * 0.015)
            
            # 楼层影响
            if 3 <= floor <= total_floors * 0.7:
                price_per_sqm *= 1.05
            elif floor == 1 or floor == total_floors:
                price_per_sqm *= 0.95
            
            # 房屋类型影响
            if property_type == '别墅':
                price_per_sqm *= 1.5
            elif property_type == '公寓':
                price_per_sqm *= 0.9
            
            # 装修影响
            decoration_multiplier = {'毛坯': 0.9, '简装': 1.0, '精装': 1.1, '豪装': 1.25}
            price_per_sqm *= decoration_multiplier[decoration]
            
            # 经济指标影响
            price_per_sqm *= (gdp_per_capita / 60000) ** 0.3
            price_per_sqm *= (population_density / 500) ** 0.2
            
            # 添加随机噪声
            price_per_sqm *= np.random.normal(1, 0.1)
            price_per_sqm = max(3000, price_per_sqm)  # 设置最低价格
            
            # 总价
            total_price = price_per_sqm * area
            
            data.append({
                '城市': city,
                '面积': round(area, 1),
                '房龄': round(age, 1),
                '楼层': floor,
                '总楼层': total_floors,
                '距离CBD距离': round(cbd_distance, 2),
                '距离地铁距离': round(subway_distance, 2),
                '周边学校数量': school_count,
                '周边医院数量': hospital_count,
                '人均GDP': int(gdp_per_capita),
                '人口密度': int(population_density),
                '房屋类型': property_type,
                '装修情况': decoration,
                '单价': int(price_per_sqm),
                '总价': int(total_price)
            })
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def basic_analysis(self):
        """
        基础数据分析
        """
        print("=" * 60)
        print("广东省房价数据基础分析")
        print("=" * 60)
        
        print(f"数据集大小: {self.data.shape}")
        print(f"数据时间跨度: 模拟数据")
        print()
        
        # 基础统计信息
        print("房价统计信息:")
        print(self.data[['单价', '总价', '面积']].describe())
        print()
        
        # 各城市平均房价
        city_price = self.data.groupby('城市').agg({
            '单价': 'mean',
            '总价': 'mean',
            '面积': 'mean'
        }).round(2)
        city_price = city_price.sort_values('单价', ascending=False)
        print("各城市平均房价排名:")
        print(city_price.head(10))
        print()
        
        # 房屋类型分布
        print("房屋类型分布:")
        print(self.data['房屋类型'].value_counts())
        print()
        
        # 装修情况分布
        print("装修情况分布:")
        print(self.data['装修情况'].value_counts())
    
    def correlation_analysis(self):
        """
        相关性分析
        """
        # 数值特征相关性
        numeric_cols = ['面积', '房龄', '楼层', '总楼层', '距离CBD距离', '距离地铁距离',
                       '周边学校数量', '周边医院数量', '人均GDP', '人口密度', '单价', '总价']
        
        correlation_matrix = self.data[numeric_cols].corr()
        
        # 找出与房价相关性最高的因素
        price_correlation = correlation_matrix['单价'].abs().sort_values(ascending=False)
        print("与房价相关性分析:")
        print(price_correlation.drop('单价'))
        
        return correlation_matrix
    
    def feature_importance_analysis(self):
        """
        特征重要性分析
        """
        # 准备数据
        features_for_model = self.data.copy()
        
        # 对分类变量进行编码
        features_for_model = pd.get_dummies(features_for_model, 
                                          columns=['城市', '房屋类型', '装修情况'])
        
        # 特征和目标变量
        feature_cols = [col for col in features_for_model.columns if col not in ['单价', '总价']]
        X = features_for_model[feature_cols]
        y = features_for_model['单价']
        
        # 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("特征重要性排名:")
        print(feature_importance.head(15))
        
        return feature_importance
    
    def create_visualizations(self):
        """
        创建数据可视化 - 分为3个窗口，每个窗口4个图
        """
        # 第一个窗口：基础房价分析
        fig1 = plt.figure(figsize=(16, 12))
        fig1.suptitle('广东省房价基础分析', fontsize=16, fontweight='bold')
        
        # 1.1 各城市平均房价
        plt.subplot(2, 2, 1)
        city_avg_price = self.data.groupby('城市')['单价'].mean().sort_values(ascending=False).head(10)
        city_avg_price.plot(kind='bar', color='skyblue')
        plt.title('前10城市平均房价', fontsize=12, fontweight='bold')
        plt.xlabel('城市')
        plt.ylabel('平均单价 (元/平方米)')
        plt.xticks(rotation=45)
        
        # 1.2 房价分布直方图
        plt.subplot(2, 2, 2)
        plt.hist(self.data['单价'], bins=50, color='lightgreen', alpha=0.7)
        plt.title('房价分布', fontsize=12, fontweight='bold')
        plt.xlabel('单价 (元/平方米)')
        plt.ylabel('频数')
        
        # 1.3 房屋类型价格分布
        plt.subplot(2, 2, 3)
        self.data.boxplot(column='单价', by='房屋类型', ax=plt.gca())
        plt.title('不同房屋类型价格分布', fontsize=12, fontweight='bold')
        plt.suptitle('')
        
        # 1.4 装修情况价格分布
        plt.subplot(2, 2, 4)
        self.data.boxplot(column='单价', by='装修情况', ax=plt.gca())
        plt.title('不同装修情况价格分布', fontsize=12, fontweight='bold')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_房价基础分析.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 第二个窗口：房屋特征与房价关系
        fig2 = plt.figure(figsize=(16, 12))
        fig2.suptitle('房屋特征与房价关系分析', fontsize=16, fontweight='bold')
        
        # 2.1 面积vs房价散点图
        plt.subplot(2, 2, 1)
        plt.scatter(self.data['面积'], self.data['单价'], alpha=0.6, color='coral')
        plt.title('面积与房价关系', fontsize=12, fontweight='bold')
        plt.xlabel('面积 (平方米)')
        plt.ylabel('单价 (元/平方米)')
        
        # 2.2 房龄vs房价
        plt.subplot(2, 2, 2)
        plt.scatter(self.data['房龄'], self.data['单价'], alpha=0.6, color='purple')
        plt.title('房龄与房价关系', fontsize=12, fontweight='bold')
        plt.xlabel('房龄 (年)')
        plt.ylabel('单价 (元/平方米)')
        
        # 2.3 楼层分布
        plt.subplot(2, 2, 3)
        floor_price = self.data.groupby('楼层')['单价'].mean()
        floor_price.plot(kind='line', marker='o', color='red')
        plt.title('不同楼层平均房价', fontsize=12, fontweight='bold')
        plt.xlabel('楼层')
        plt.ylabel('平均单价 (元/平方米)')
        
        # 2.4 人均GDP vs 房价
        plt.subplot(2, 2, 4)
        plt.scatter(self.data['人均GDP'], self.data['单价'], alpha=0.6, color='green')
        plt.title('人均GDP与房价关系', fontsize=12, fontweight='bold')
        plt.xlabel('人均GDP (元)')
        plt.ylabel('单价 (元/平方米)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_房屋特征与房价关系.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 第三个窗口：区位因素与相关性分析
        fig3 = plt.figure(figsize=(16, 12))
        fig3.suptitle('区位因素与相关性分析', fontsize=16, fontweight='bold')
        
        # 3.1 距离CBD距离vs房价
        plt.subplot(2, 2, 1)
        plt.scatter(self.data['距离CBD距离'], self.data['单价'], alpha=0.6, color='orange')
        plt.title('距离CBD距离与房价关系', fontsize=12, fontweight='bold')
        plt.xlabel('距离CBD距离 (公里)')
        plt.ylabel('单价 (元/平方米)')
        
        # 3.2 周边设施vs房价
        plt.subplot(2, 2, 2)
        plt.scatter(self.data['周边学校数量'], self.data['单价'], 
                   alpha=0.6, color='blue', label='学校', s=30)
        plt.scatter(self.data['周边医院数量'], self.data['单价'], 
                   alpha=0.6, color='red', label='医院', s=30)
        plt.title('周边设施与房价关系', fontsize=12, fontweight='bold')
        plt.xlabel('设施数量')
        plt.ylabel('单价 (元/平方米)')
        plt.legend()
        
        # 3.3 相关性热力图
        plt.subplot(2, 2, 3)
        numeric_cols = ['面积', '房龄', '距离CBD距离', '距离地铁距离', '周边学校数量', '单价']
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图', fontsize=12, fontweight='bold')
        
        # 3.4 主要城市房价箱线图
        plt.subplot(2, 2, 4)
        top_cities = self.data.groupby('城市')['单价'].mean().nlargest(8).index
        data_top_cities = self.data[self.data['城市'].isin(top_cities)]
        data_top_cities.boxplot(column='单价', by='城市', ax=plt.gca())
        plt.title('主要城市房价分布', fontsize=12, fontweight='bold')
        plt.suptitle('')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_区位因素与相关性分析.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def predictive_modeling(self):
        """
        建立预测模型
        """
        print("=" * 60)
        print("房价预测模型")
        print("=" * 60)
        
        # 准备数据
        model_data = self.data.copy()
        model_data = pd.get_dummies(model_data, columns=['城市', '房屋类型', '装修情况'])
        
        # 特征和目标变量
        feature_cols = [col for col in model_data.columns if col not in ['单价', '总价']]
        X = model_data[feature_cols]
        y = model_data['单价']
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 线性回归模型
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # 随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # 模型评估
        print("线性回归模型:")
        print(f"R² Score: {r2_score(y_test, lr_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, lr_pred):.2f}")
        print()
        
        print("随机森林模型:")
        print(f"R² Score: {r2_score(y_test, rf_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}")
        
        # 创建模型评估可视化
        fig4 = plt.figure(figsize=(16, 12))
        fig4.suptitle('模型预测效果评估', fontsize=16, fontweight='bold')
        
        # 4.1 线性回归预测vs实际
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, lr_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'线性回归模型 (R²={r2_score(y_test, lr_pred):.3f})', fontsize=12, fontweight='bold')
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        
        # 4.2 随机森林预测vs实际
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, rf_pred, alpha=0.6, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'随机森林模型 (R²={r2_score(y_test, rf_pred):.3f})', fontsize=12, fontweight='bold')
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        
        # 4.3 特征重要性图
        plt.subplot(2, 2, 3)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.title('特征重要性排名(前10)', fontsize=12, fontweight='bold')
        plt.xlabel('重要性')
        
        # 4.4 残差分析
        plt.subplot(2, 2, 4)
        residuals = y_test - rf_pred
        plt.scatter(rf_pred, residuals, alpha=0.6, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title('随机森林模型残差分析', fontsize=12, fontweight='bold')
        plt.xlabel('预测价格')
        plt.ylabel('残差')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_模型预测效果评估.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return lr_model, rf_model, scaler
    
    def generate_insights(self):
        """
        生成分析洞察
        """
        print("=" * 60)
        print("广东省房价分析洞察")
        print("=" * 60)
        
        # 价格分析
        avg_price = self.data['单价'].mean()
        max_city = self.data.groupby('城市')['单价'].mean().idxmax()
        min_city = self.data.groupby('城市')['单价'].mean().idxmin()
        
        print(f"1. 价格概况:")
        print(f"   - 全省平均房价: {avg_price:,.0f} 元/平方米")
        print(f"   - 最高房价城市: {max_city}")
        print(f"   - 最低房价城市: {min_city}")
        print()
        
        # 面积分析
        avg_area = self.data['面积'].mean()
        large_house_ratio = (self.data['面积'] > 120).mean() * 100
        
        print(f"2. 房屋面积:")
        print(f"   - 平均面积: {avg_area:.1f} 平方米")
        print(f"   - 大户型比例(>120平): {large_house_ratio:.1f}%")
        print()
        
        # 区位分析
        avg_cbd_distance = self.data['距离CBD距离'].mean()
        close_cbd_premium = self.data[self.data['距离CBD距离'] < 5]['单价'].mean() / avg_price - 1
        
        print(f"3. 区位因素:")
        print(f"   - 平均距CBD距离: {avg_cbd_distance:.1f} 公里")
        print(f"   - 市中心房价溢价: {close_cbd_premium:.1%}")
        print()
        
        # 房龄影响
        new_house_premium = self.data[self.data['房龄'] < 5]['单价'].mean() / avg_price - 1
        old_house_discount = 1 - self.data[self.data['房龄'] > 15]['单价'].mean() / avg_price
        
        print(f"4. 房龄影响:")
        print(f"   - 新房溢价(<5年): {new_house_premium:.1%}")
        print(f"   - 老房折价(>15年): {old_house_discount:.1%}")
        print()
        
        print("5. 投资建议:")
        print("   - 深圳、广州等一线城市仍有较高投资价值")
        print("   - 关注距离CBD较近的区域")
        print("   - 优质学区房具有保值增值潜力")
        print("   - 新房比老房具有明显价格优势")
    
    def run_complete_analysis(self):
        """
        运行完整分析流程
        """
        print("开始广东省房价影响因素分析...")
        print()
        
        # 1. 生成数据
        print("正在生成模拟数据...")
        self.generate_mock_data()
        print("数据生成完成!")
        print()
        
        # 2. 基础分析
        self.basic_analysis()
        print()
        
        # 3. 相关性分析
        print("正在进行相关性分析...")
        correlation_matrix = self.correlation_analysis()
        print()
        
        # 4. 特征重要性分析
        print("正在进行特征重要性分析...")
        feature_importance = self.feature_importance_analysis()
        print()
        
        # 5. 可视化
        print("正在生成可视化图表...")
        self.create_visualizations()
        print()
        
        # 6. 预测模型
        print("正在建立预测模型...")
        lr_model, rf_model, scaler = self.predictive_modeling()
        print()
        
        # 7. 生成洞察
        self.generate_insights()
        
        print(f"\n分析完成!")
        print(f"所有图片已保存到 '{self.output_dir}' 文件夹中:")
        print("  - 01_房价基础分析.png")
        print("  - 02_房屋特征与房价关系.png")
        print("  - 03_区位因素与相关性分析.png")
        print("  - 04_模型预测效果评估.png")
        
        return {
            'data': self.data,
            'correlation_matrix': correlation_matrix,
            'feature_importance': feature_importance,
            'models': {'linear_regression': lr_model, 'random_forest': rf_model},
            'scaler': scaler
        }

# 运行分析
if __name__ == "__main__":
    # 创建分析实例
    analyzer = GuangdongHousingAnalysis()
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    # 保存数据
    analyzer.data.to_csv('广东省房价数据.csv', index=False, encoding='utf-8')
    print("\n数据已保存至 '广东省房价数据.csv'")