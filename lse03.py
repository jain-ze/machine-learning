#多項式回歸 Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt

#1 隨機產生二次方程式資料
np.random.seed(1)
m = 1000
x1 = 10 * np.random.rand(m, 1) - 6 # x1 = 10 * random number between 0 and 1 - 6
y = 10 + 6 * x1 + 5 *x1**2 + 30 * np.random.randn(m, 1) # y = 10 + 6x1 + 5x1^2 + noise

#2 繪製原始資料圖形
plt.figure(figsize=(8, 6))
plt.plot(x1, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.title('Raw Data Distribution Chart', fontsize=18)# 設定標題
plt.savefig("lse03_1.png")
plt.savefig("lse03_1.pdf")
plt.show()

#3 資料前處理
#Future Polynomial 轉換 degree=2
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False) #創建一個 PolynomialFeatures 對象，指定二次多項式(degree=2)和不包括偏差項(即不包括截距)
x1_p = poly_features.fit_transform(x1)#對 x1 進行二次多項式轉換
#print(x1[0], x1_p[0])#顯示第1筆原始資料和轉換後的資料

#4 Future Scaling標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()#創建一個 StandardScaler 對象
x1_ps = scaler.fit_transform(x1_p)#對 x1_p 進行標準化

#5 配適模型
#5-1 使用線性回歸模型LinearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()#創建一個 LinearRegression 對象
lin_reg.fit(x1_ps, y)#對標準化後的資料進行擬合
#print(lin_reg.intercept_, lin_reg.coef_)#顯示截距b和係數w1, w2
#5-2 使用梯度下降法SGDRegressor
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)#創建一個 SGDRegressor 對象
sgd_reg.fit(x1_ps, y.ravel())#對標準化後的資料進行擬合，ravel()函數將 y 轉換為一維數組
#print(sgd_reg.intercept_, sgd_reg.coef_)#顯示截距b和係數w1, w2

#6 預測Predict
x1_new = [[-5], [1]]#新的 x1 值，假設有兩筆資料-5和1
x1_new_p = poly_features.transform(x1_new)#對新的 x1 值進行二次多項式轉換
x1_new_ps = scaler.transform(x1_new_p)#對新的 x1 值進行標準化
y_new_lin = lin_reg.predict(x1_new_ps)#使用線性回歸模型進行預測
y_new_sgd = sgd_reg.predict(x1_new_ps)#使用梯度下降法進行預測
#print(y_new_lin, y_new_sgd)#顯示線性回歸模型和梯度下降法的預測結果

#7 繪製模型圖形
plt.figure(figsize=(8, 6))
plt.plot(x1, y, "b.")#繪製原始資料
x1s = np.linspace(x1.min(), x1.max(), 100).reshape(-1, 1)#產生100個 x1 值
x1s_p = poly_features.transform(x1s)#對 x1s 進行二次多項式轉換
x1s_ps = scaler.transform(x1s_p)#對 x1s_p 進行標準化
y_predict_lin = lin_reg.predict(x1s_ps)#使用線性回歸模型進行預測
y_predict_sgd = sgd_reg.predict(x1s_ps)#使用梯度下降法進行預測
lin_w1, lin_w2 = lin_reg.coef_[0]#取得線性回歸模型的係數
lin_b = lin_reg.intercept_[0]  # 提取單個元素
plt.plot(x1s, y_predict_lin, "r-", linewidth=2, label="Linear Regression:$\\hat y$= %.2f + %.2fx1 + %.2fx1^2" % (lin_b, lin_w1, lin_w2) )#繪製線性回歸模型
sgd_w1, sgd_w2 = sgd_reg.coef_[0], sgd_reg.coef_[1]#取得梯度下降法的係數
sgd_b = sgd_reg.intercept_[0]
plt.plot(x1s, y_predict_sgd, "g--", linewidth=3, label="SGD Regression:$\\hat y$= %.2f + %.2fx1 + %.2fx1^2" % (sgd_b, sgd_w1, sgd_w2))#繪製梯度下降法模型
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=12)
plt.savefig("lse03_2.png")
plt.savefig("lse03_2.pdf")
plt.show()

