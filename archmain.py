import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 字体管理器

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=5)  # 设置汉字格式

# 导入训练数据集和测试集
train_data = pd.read_csv('happiness_train_complete.csv', encoding='gbk')
test_data = pd.read_csv('happiness_test_complete.csv', encoding='gbk')
# print(train_data)

#数据初步可视化step1
x = train_data.id
y = train_data.happiness
plt.figure(figsize=(10,6))
plt.title(u"幸福感随id分布",fontproperties=font)
plt.scatter(x, y,c='#DC143C')
plt.xlabel("id",fontsize=18)
plt.ylabel(u'幸福感值',fontsize=18,fontproperties=font)
plt.show()
#数据初步可视化step2
tmp=train_data.happiness
k=0
res={}
for index,k in enumerate(tmp):
    if k not in res:
        res[k]=1
    else:
        res[k]+=1
#print(res)
x=[];y=[]
for k in res:
    x.append(k)
    y.append(res[k])
plt.title(u"幸福感数量分布", fontproperties=font)
plt.xlabel("幸福感值",fontsize=18,fontproperties=font)
plt.ylabel(u'数量',fontsize=18,fontproperties=font)
plt.bar(x,y)
plt.show()

# 数据初步处理
train_data_y = train_data.happiness
# print(train_data_y)
# 删除含有y值的列
train_data.drop(["happiness"], axis=1, inplace=True)
# 合并训练集和测试集
data = pd.concat((train_data, test_data), axis=0)

# 调查时间是对幸福感影响不大，故删掉
data.drop("survey_time", axis=1, inplace=True)
# print(data)

# 首先处理特征，首先获取每个特征缺失的情况
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
tmp = percent.to_dict()
# print(tmp)

#可视化缺失率
tmp1=[]
tmp2=[]
for index,k in enumerate(percent):
    tmp1.append(index)
    tmp2.append(k)
plt.ylabel(u"缺失率",fontproperties=font)
plt.xlabel(u"特征",fontproperties=font)
plt.bar(tmp1,tmp2)
plt.show()

# 获取缺失率大于50％的特征
tmp1 = [];
tmp2 = []
for k in tmp:
    if float(tmp[k]) > 0.5:
        tmp1.append(k)
        tmp2.append(float(tmp[k]))
# print(tmp1)

#可视化缺失率大于50％的特征
plt.ylabel(u"缺失率",fontproperties=font)
plt.xlabel(u"特征",fontproperties=font)
plt.bar(tmp1,tmp2)
plt.show()

# 由于缺失率过高，因此删除确实率大于50％的特征
data.drop(tmp1, axis=1, inplace=True)
# print(data)

#当去除一部分缺失率过大的特征之后，开始处理缺失率并不高的某些特征，对其进行填充
#打印仍然有缺失值的特征
tmp1=[];tmp2=[]
for k in tmp:
    if float(tmp[k])<0.5 and float(tmp[k])>0:
        tmp1.append(k)
        tmp2.append(float(tmp[k]))
#print(tmp1)

# 观察到marital_now以及marital_1st的空缺可能是由于未结婚造成的，填充为9997
# print("marital_now",data.marital_now.isnull().sum())
data.marital_now.fillna(9997, inplace=True)
# print("marital_1st",data.marital_1st.isnull().sum())
data.marital_1st.fillna(9997, inplace=True)
# 同样，s_xxx这一类特征，都是关于被调查人配偶的情况，也可能是由于被调查人可能没有配偶而导致该项缺失，因此将该项用0填补。
# print("s_political",data.s_political.isnull().sum())
data.s_political.fillna(0, inplace=True)
# print("s_hukou",data.s_hukou.isnull().sum())
data.s_hukou.fillna(0, inplace=True)
# print("s_income",data.s_income.isnull().sum())
data.s_income.fillna(0, inplace=True)
# print("s_birth",data.s_birth.isnull().sum())
data.s_birth.fillna(0, inplace=True)
# print("s_edu",data.s_edu.isnull().sum())
data.s_edu.fillna(0, inplace=True)
# print("s_work_exper",data.s_work_exper.isnull().sum())
data.s_work_exper.fillna(0, inplace=True)
# minor_child空缺可能是因为没有孩子，填充为0
# print("minor_child",data.minor_child.isnull().sum())
data.minor_child.fillna(0, inplace=True)
# 根据输出可以看出，family_income这一特征只有1次缺失，可能是由于被调察人的疏忽造成的，因此将此项填写为family_income的众数
# print("family_income",data.family_income.isnull().sum())
data.family_income.fillna(data.family_income.mode(), inplace=True)
# 打印测试：全部已经填充完毕，已经没有缺失项。
# print(data.isnull().sum()>0)

# 然后处理标签，当幸福感为-8时，被调查者无法回答自身的幸福感值，考虑到概率问题，将-8设置为众数4
train_data_y = train_data_y.map(lambda x: 4 if x == -8 else x)
# print(train_data_y)

#可视化：初步了解年代的分层情况
x=data.id
y=data.birth
plt.scatter(x,y)
plt.show()


# 对于年代进行泛化
# 出生的年代
def year(x):
    if (x == 0):
        return 0
    elif (0 < x < 1920):
        return 1
    elif 1920 <= x <= 1930:
        return 2
    elif 1930 < x <= 1940:
        return 3
    elif 1940 < x <= 1950:
        return 4
    elif 1950 < x <= 1960:
        return 5
    elif 1960 < x <= 1970:
        return 6
    elif 1970 < x <= 1980:
        return 7
    elif 1980 < x <= 1990:
        return 8
    elif 1990 < x <= 2000:
        return 9
    elif 2000 < x:
        return 10


# 自己出生年代
data["birth"] = data["birth"].map(year)
# 配偶出生年代
data["s_birth"] = data["s_birth"].map(year)
# 父亲出生年代
data["f_birth"] = data["f_birth"].map(year)
# 母亲出生年代
data["m_birth"] = data["m_birth"].map(year)

# 使用零-均值规范化方法，进行均值归一化处理
# 对于被调查者自己输入的值，并不是由问卷给出，可能有若干不同的情况，因此将其规范化
num_col = ['income', 'floor_area', 'height_cm', 'weight_jin', 'family_income', 'family_m', 'house', 'son', 'daughter',
           'minor_child', 's_income',
           'inc_exp', 'public_service_1', 'public_service_2', 'public_service_3', 'public_service_4',
           'public_service_5', 'public_service_6', 'public_service_7', 'public_service_8', 'public_service_9']
num_col_std = data.loc[:, num_col].std()
# print(num_col_std)
num_col_mean = data.loc[:, num_col].mean()
# print(num_col_mean)
num_data = (data.loc[:, num_col] - num_col_mean) / num_col_std
# print(num_data)

# 对其他离散类型进行规范化,使用one-hot编码
other_data = data.drop(num_col, axis=1)
other_data = other_data.astype(str)
# print(other_data)
for cols in list(other_data.iloc[:, 1:].columns):
    other_data = pd.get_dummies(other_data.iloc[:, 1:], prefix=cols)

# 合并数值和离散特征
data = pd.concat((other_data, num_data), axis=1)
# print(data)

# 标准化之后重新分出训练集和测试集
train_data = data.iloc[:8000, :]
train_data_x = train_data.values
test_data = data.iloc[8000:, :]
test_data_x = test_data.values

# 开始进行训练，这里使用xgboost算法,并且结合交叉验证
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

xgb_mat = np.zeros(8000)
xgb_pred = np.zeros(2968)

'''''
xgboost参数设定：
booster使用基于树的模型
eta=0.01 通过缩减特征的权重使提升计算过程更加保守，防止过拟合
max_depth=5 树的最大深度，树的深度越大，则对数据的拟合程度越高
subsample=0.7 用于训练模型的子样本占整个样本集合的比例，能够防止过拟合
colsample_bytree=0.6 在建立树时对特征随机采样的比例
'''''
xgb_params = {'eta': 0.01, 'max_depth': 5, 'subsample': 0.5, 'colsample_bytree': 0.3}

'''''
Cross-validatio参数设定：
n_splits=5 进行5折交叉验证，经试验：10折交叉验证相较于5折交叉验证提升效果并不明显
shuffle=True 每次生成随机数据
random_state=17 随机种子
'''''
cross_val = KFold(n_splits=5, shuffle=True, random_state=17)

for k, (train_index, value_index) in enumerate(cross_val.split(train_data_x, train_data_y)):
    train_tmp = xgb.DMatrix(train_data_x[train_index], train_data_y[train_index])
    value_tmp = xgb.DMatrix(train_data_x[value_index], train_data_y[value_index])
    print("Cross-validatio: {}".format(k + 1))
    watchlist = [(train_tmp, 'train'), (value_tmp, 'valid_data')]
    w = xgb.train(params=xgb_params, dtrain=train_tmp, num_boost_round=3000, evals=watchlist,
                  early_stopping_rounds=100, verbose_eval=200)
    xgb_mat[value_index] = w.predict(xgb.DMatrix(train_data_x[value_index]))
    xgb_pred += w.predict(xgb.DMatrix(test_data_x)) / cross_val.n_splits

print("Local test score: {:0.4f}".format(mean_squared_error(xgb_mat, train_data_y)))
# print(xgb_pred)

# 预测结束，将预测值写入happiness_submit.csv文件
submit_data = pd.read_csv("happiness_submit.csv", encoding='gbk')
submit_data["happiness"] = list(xgb_pred)
submit_data.to_csv("happiness_submit.csv", index=False)