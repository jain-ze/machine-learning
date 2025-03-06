#最小平方法

# 引入套件
import numpy as np # 引入NumPy
import matplotlib.pyplot as plt # 引入Matplotlib
from sklearn.linear_model import LinearRegression   # 引入線性迴歸模型
from sklearn.metrics import mean_squared_error # 引入均方誤差

# 隨機產生資料
np.random.seed(1)#設定隨機種子1，確保每次執行結果相同
m = 100
x1 = 50 + 30 * np.random.rand(m, 1) # 產生 100 個 50~80 之間的亂數，有點像是體重
y = 135 + 0.5 * x1 + 3 * np.random.randn(m, 1) # 產生 100 個 y = 135 + 0.5x1 + 高斯雜訊，有點像是身高
# 繪製資料
plt.figure(figsize=(5, 4))# 設定圖片大小
plt.plot(x1, y, "b.")# 繪製原始資料以藍色點點表示
plt.xlabel("$x_1$", fontsize=18)# 設定x軸標籤
plt.ylabel("$y$", rotation=0, fontsize=18)# 設定y軸標籤
plt.title('Raw Data Distribution Chart', fontsize=18)# 設定標題
#plt.axis([30, 100, 130, 200]) # 設定座標軸範圍
plt.savefig('fig1.png') # 儲存圖片
plt.savefig('plot_ex1.pdf' , format='pdf', dpi=300, bbox_inches='tight') # 儲存成PDF
plt.show() # 顯示圖片

# 適配模型
lin_reg = LinearRegression() # 創建線性迴歸模型
lin_reg.fit(x1, y) # 使用最小平方法配適模型
#print(lin_reg.intercept_, lin_reg.coef_) # 模型的參數，分別是截距b和斜率w1
# 繪製模型
#x1_new = np.array([[50],[60],[70], [80]]) # 產生30~100之間的數字
#y_predict = lin_reg.predict(x1_new) # 使用模型進行預測
#print(y_predict)#印出預測值
x1s = np.linspace(x1.min(), x1.max(), 10).reshape(-1, 1) # 由x1的最小值到最大值等距產生10個數字
y_predict = lin_reg.predict(x1s) # 使用模型進行預測
plt.figure(figsize=(5, 4)) # 設定圖片大小
plt.plot(x1s, y_predict, "r-", linewidth=2, label="$\hat y$") # 繪製預測結果以紅色實線表示
plt.plot(x1, y, "b.") # 繪製原始資料以藍色點點表示
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14) # 顯示圖例
#計算模型的參數
y_predict = lin_reg.predict(x1) # 使用模型進行預測
lin_mse = mean_squared_error(y, y_predict) # 計算均方誤差
lin_rmse = np.sqrt(lin_mse) # 計算均方根誤差
#在圖例印出均方根誤差及模型函式
plt.text(50, 180, 'RMSE=%.2f' % lin_rmse, fontsize=12)
plt.text(65, 180, '$\hat y$= %.2f + %.2fx' % (lin_reg.intercept_, lin_reg.coef_), fontsize=12)
#plt.title('RMSE=%.2f' % lin_rmse)
#print(lin_rmse) # 印出均方根誤差
plt.savefig('fig2.png') # 儲存圖片
plt.savefig('plot_ex2.pdf', format='pdf', dpi=300, bbox_inches='tight')  # 儲存成PDF
plt.show()


