#梯度下降法

# 引入套件
import numpy as np # 引入NumPy
import matplotlib.pyplot as plt # 引入Matplotlib
from sklearn.preprocessing import StandardScaler # 引入標準化套件
from sklearn.linear_model import SGDRegressor # 引入梯度下降法套件(S表示隨機梯度下降法)
from sklearn.metrics import mean_squared_error # 引入均方誤差


# 隨機產生資料
np.random.seed(1)#設定隨機種子1，確保每次執行結果相同
m = 100
x1 = 50 + 30 * np.random.rand(m, 1) # 產生 100 個 50~80 之間的亂數，有點像是體重
y = 135 + 0.5 * x1 + 3 * np.random.randn(m, 1) # 產生 100 個 y = 135 + 0.5x1 + 高斯雜訊，有點像是身高
#Feature 標準化
scal = StandardScaler() # 創建標準化物件
x1_scal = scal.fit_transform(x1) # 對x1進行標準化(標準常態轉換，fit會算出這個feature的平均值及標準差，transform會將所有的featurer減掉平均值再除以標準差)
#print(x1_scal[:5]) # 印出前5筆資料
#print(scal.mean_, scal.scale_) # 印出平均值及標準差
#繪製資料
plt.figure(figsize=(5, 4))# 設定圖片大小
plt.plot(x1_scal, y, "b.")# 繪製原始資料以藍色點點表示
plt.xlabel("$x_1$", fontsize=18)# 設定x軸標籤   
plt.ylabel("$y$", rotation=0, fontsize=18)# 設定y軸標籤
plt.title('Raw Data Distribution Chart', fontsize=18)# 設定標題
#plt.axis([30, 100, 130, 200]) # 設定座標軸範圍
plt.savefig('fig3.png') # 儲存圖片
plt.savefig('plot_ex3.pdf' , format='pdf', dpi=300, bbox_inches='tight') # 儲存成PDF
plt.show() # 顯示圖片
#適配模型
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) # 創建梯度下降法模型，max_iter是最大迭代次數，tol是容忍度，penalty是正則化項，eta0是學習速率
sgd_reg.fit(x1_scal, y.ravel()) # 使用梯度下降法配適模型，ravel()是將y轉換成一維陣列
#print(sgd_reg.intercept_, sgd_reg.coef_) # 印出截距b和斜率w1
#進行預測predict
x1_new = np.array([[50],[60],[70], [80], [85], [99], [120]]) # 產生30~100之間的數字
x1_new_scal = scal.transform(x1_new) # 對x1_new進行標準化
y_predict = sgd_reg.predict(x1_new_scal) # 使用模型進行預測
#print(y_predict)#印出預測值

#繪製模型
x1s = np.linspace(x1_scal.min(), x1_scal.max(), 10).reshape(-1, 1) # 由x1的最小值到最大值等距產生10個數字
y_predict = sgd_reg.predict(x1s) # 使用模型進行預測
plt.figure(figsize=(5, 4)) # 設定圖片大小
plt.plot(x1s, y_predict, "r-", linewidth=2, label="$\hat y$") # 繪製預測結果以紅色實線表示
plt.plot(x1_scal, y, "b.") # 繪製原始資料以藍色點點表示
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14) # 顯示圖例
#計算模型的參數
y_predict = sgd_reg.predict(x1_scal) # 使用模型進行預測
lin_mse = mean_squared_error(y, y_predict) # 計算均方誤差
lin_rmse = np.sqrt(lin_mse) # 計算均方根誤差
#在圖例印出均方根誤差及模型函式
plt.text(-1.5, 180, 'RMSE=%.2f' % lin_rmse, fontsize=12)
plt.text(0, 180, '$\hat y$= %.2f + %.2fx' % (sgd_reg.intercept_, sgd_reg.coef_), fontsize=12)
plt.savefig('fig4.png') # 儲存圖片  
plt.savefig('plot_ex4.pdf', format='pdf', dpi=300, bbox_inches='tight')  # 儲存成PDF
plt.show()




#手動計算批次梯度下降法
'''
#X = np.c_[np.ones((m, 1)), x1_scal] # 在x1_scal前加入一個全為1的column
#eta = 0.1 # 學習速率
#n_iterations = 1000 # 迭代次數
#m = 100 # 資料數
#theta = np.random.randn(2,1) # 隨機初始化參數
#for iteration in range(n_iterations):
#    gradients = 2/m * X.T.dot(X.dot(theta) - y) # 計算梯度
#    theta = theta - eta * gradients # 更新參數
#    if iteration % 10 == 0:
#        print(theta) # 印出參數
'''

#手動計算隨機梯度下降法
'''
n_epochs = 50 # 迭代次數
theta = np.random.randn(2,1) # 隨機初始化參數，theta是一個2x1的矩陣，裡面的值是隨機產生的，代表截距b和斜率w1
t0, t1 = 5, 50 # 學習速率的超參數，t0和t1是隨機設定的，可以自行調整，這兩個參數會影響學習速率，t0是分子，t1是分母
def learning_schedule(t): # 學習速率的函數
    return t0 / (t + t1) # 學習速率會隨著迭代次數增加而遞減，t0是分子，t1是分母，t是迭代次數
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m) # 隨機選取一筆資料
        xi = x1_scal[random_index:random_index+1] # 選取x1_scal的資料
        xi = np.c_[np.ones((1, 1)), xi] # 在xi前加入一個全為1的column，因為theta是一個2x1的矩陣，所以要加入一個1，讓xi變成1x2的矩陣，才能和theta相乘
        yi = y[random_index:random_index+1] # 選取y的資料
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) # 計算梯度
        eta = learning_schedule(epoch * m + i) # 計算學習速率，學習速率會隨著迭代次數增加而遞減
        theta = theta - eta * gradients # 更新參數
        if i % 100 == 0:
            print(theta) # 印出參數
'''
#手動計算小批次梯度下降法
'''
n_iterations = 50 # 迭代次數
minibatch_size = 20 # 小批次的大小
theta = np.random.randn(2,1) # 隨機初始化參數
t = 0
t0, t1 = 200, 1000 # 學習速率的超參數
def learning_schedule(t): # 學習速率的函數
    return t0 / (t + t1) # 學習速率會隨著迭代次數增加而遞減 
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m) # 隨機打亂資料  # permutation是將0~m-1的數字打亂
    x1_shuffled = x1_scal[shuffled_indices] # 打亂x1_scal的資料
    y_shuffled = y[shuffled_indices] # 打亂y的資料
    for i in range(0, m, minibatch_size):
        t += 1
        xi = x1_shuffled[i:i+minibatch_size] # 選取x1_shuffled的資料
        xi = np.c_[np.ones((minibatch_size, 1)), xi] # 在xi前加入一個全為1的column
        yi = y_shuffled[i:i+minibatch_size] # 選取y_shuffled的資料
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi) # 計算梯度
        eta = learning_schedule(t) # 計算學習速率
        theta = theta - eta * gradients # 更新參數
        if i % 100 == 0:
            print(theta) # 印出參數
'''

