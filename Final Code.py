
# In[1]:

# 导入数据分析及可视化的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# In[2]:
# 读取数据
data_train = pd.read_csv('D:/NewDesktop/data/train.csv')
data_testA = pd.read_csv('D:/NewDesktop/data/testA.csv')

# In[3]:

# 查看数据集初始状态
data_train.head()
# In[4]:
data_testA.head()
# In[5]:
# 查看数据集总体分布
data_train.shape
data_testA.shape
data_train.info()    #通过info()查看特征的数据类型
data_testA.info()
data_train.describe()    #粗略查看各特征的基本统计量
data_testA.describe()    #粗略查看data_test_a的基本统计量
# In[6]:
# 正负样本比
plt.hist(data_train['isDefault'])
plt.title("positive vs negative")
plt.show()

# train 中 nan缺失值的可视化
missing_features_train = data_train.isnull().sum()
missing_features_train = missing_features_train[missing_features_train > 0]
num_missing_features_train = len(missing_features_train)

missing_features_test = data_testA.isnull().sum()
missing_features_test = missing_features_test[missing_features_test > 0]
num_missing_features_test = len(missing_features_test)

# Visualization
plt.figure(figsize=(15, 6))

# Train data
plt.subplot(1, 2, 1)
sns.barplot(x=missing_features_train.index, y=missing_features_train.values)
plt.xticks(rotation=90)
plt.title(f'Train Data - Number of Missing Features: {num_missing_features_train}')
plt.ylabel('Number of Missing Values')
plt.xlabel('Feature Names')

# Test data
plt.subplot(1, 2, 2)
sns.barplot(x=missing_features_test.index, y=missing_features_test.values)
plt.xticks(rotation=90)
plt.title(f'Test Data - Number of Missing Features: {num_missing_features_test}')
plt.ylabel('Number of Missing Values')
plt.xlabel('Feature Names')

plt.tight_layout()
plt.show()

num_missing_features_train, num_missing_features_test
# In[7]:
# 查看缺失值大于50%的特征数量：无
have_null_fea_dict = (data_train.isnull().sum()/len(data_train)).to_dict()
fea_null_moreThanHalf = {}
for key,value in have_null_fea_dict.items():
    if value > 0.5:
        fea_null_moreThanHalf[key] = value
fea_null_moreThanHalf

# 离散数据分布
# Grade
data_train['grade'].value_counts().sort_index().plot.bar()
# subGrade
data_train['subGrade'].value_counts().sort_index().plot.bar(figsize=(15, 5))
# In[8]:
# 查看训练集测试集中特征属性只有一值的特征
one_value_fea = [col for col in data_train.columns if data_train[col].nunique() <= 1]
one_value_fea_test = [col for col in data_testA.columns if data_testA[col].nunique() <= 1]
one_value_fea
one_value_fea_test
# In[9]:

# 将特征分为类别型特征和数值型特征，即：字符串型和数值型
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))

# In[10]:


# 划分数值型变量中的连续变量和离散型变量
def get_numerical_serial_fea(data,feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea
numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)


# In[11]:
# 发现在训练集47个特征中，有5项类别型特征，42项数值型，其中33项连续性，9项离散型特征

# In[12]:
# 对数值型连续型变量进行分析
# 可视化
f = pd.melt(data_train, value_vars=numerical_serial_fea)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

# In[13]:
# 对数值型离散型变量特征分析

# In[14]:
data_train['term'].value_counts()
# In[15]:
data_train['homeOwnership'].value_counts()
# In[16]:
data_train['verificationStatus'].value_counts()
# In[17]:
data_train['initialListStatus'].value_counts()
# In[18]:
data_train['applicationType'].value_counts()
# In[19]:
data_train['policyCode'].value_counts()#无用，全部一个值
# In[20]:
data_train['n11'].value_counts()
# In[21]:
data_train['n12'].value_counts()
# In[22]:
# 类别型变量分析
category_fea
# In[23]:
data_train['grade'].value_counts()
# In[24]:
data_train['subGrade'].value_counts()
# In[25]:
data_train['employmentLength'].value_counts()
# In[26]:
data_train['issueDate'].value_counts()
# In[27]:
data_train['earliesCreditLine'].value_counts()
# In[28]:
# 查看标签 isDefault 项
data_train['isDefault'].value_counts()
# In[29]:
# 查看employmentLength项
plt.figure(figsize=(8, 8))
sns.barplot(x=data_train["employmentLength"].value_counts(dropna=False)[:20].values, 
            y=data_train["employmentLength"].value_counts(dropna=False)[:20].keys())
plt.show()
# In[30]:
train_loan_fr = data_train.loc[data_train['isDefault'] == 1]
train_loan_nofr = data_train.loc[data_train['isDefault'] == 0]


# In[31]:

# 查看类别型变量在不同y值上的分布
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_loan_fr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade fraud')
train_loan_nofr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-fraud')
train_loan_fr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength fraud')
train_loan_nofr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-fraud')
plt.show()


# In[32]:


# 查看连续型变量在不同y值上的分布
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_loan_fr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade fraud')
train_loan_nofr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-fraud')
train_loan_fr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength fraud')
train_loan_nofr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-fraud')
plt.show()


# In[33]:

total = len(data_train)
total_amt = data_train.groupby(['isDefault'])['loanAmnt'].sum().sum()
plt.figure(figsize=(12,5))
plt.subplot(121)
plot_tr = sns.countplot(x='isDefault',data=data_train)
plot_tr.set_title("Fraud Loan Distribution \n 0: good user | 1: bad user", fontsize=14)
plot_tr.set_xlabel("Is fraud by count", fontsize=16)
plot_tr.set_ylabel('Count', fontsize=16)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 
    
percent_amt = (data_train.groupby(['isDefault'])['loanAmnt'].sum())
percent_amt = percent_amt.reset_index()
plt.subplot(122)
plot_tr_2 = sns.barplot(x='isDefault', y='loanAmnt',  dodge=True, data=percent_amt)
plot_tr_2.set_title("Total Amount in loanAmnt  \n 0: good user | 1: bad user", fontsize=14)
plot_tr_2.set_xlabel("Is fraud by percent", fontsize=16)
plot_tr_2.set_ylabel('Total Loan Amount Scalar', fontsize=16)
for p in plot_tr_2.patches:
    height = p.get_height()
    plot_tr_2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total_amt * 100),
            ha="center", fontsize=15)

# In[34]:
# 查看与时间变相关的量issueData
# issueDateDT特征表示数据日期离数据集中日期最早的日期（2007-06-01）的天数
data_train['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_train['issueDateDT'] = data_train['issueDate'].apply(lambda x: x-startdate).dt.days

data_testA['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_testA['issueDateDT'] = data_testA['issueDate'].apply(lambda x: x-startdate).dt.days

# Plot
plt.hist(data_train['issueDateDT'], bins=50, alpha=0.5, label='train')
plt.hist(data_testA['issueDateDT'], bins=50, alpha=0.5, label='test')
plt.legend()
plt.title('Distribution of issueDateDT dates')
plt.xlabel('Days since 2007-06-01')
plt.ylabel('Frequency')
plt.show()
# In[35]:
# 数据预处理
# In[36]:
# 查看缺失值情况
data_train.isnull().sum()
# In[37]:


#数据缺失值的填充，时间格式特征的转化处理，某些对象类别特征的处理
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
label = 'isDefault'
numerical_fea.remove(label)


# In[38]:


# 查看缺失值情况
data_train.isnull().sum()


# In[39]:


# 查看缺失值情况
data_testA.isnull().sum()


# In[40]:

# 按照中位数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_testA[numerical_fea] = data_testA[numerical_fea].fillna(data_train[numerical_fea].median())

# 按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_testA[category_fea] = data_testA[category_fea].fillna(data_train[category_fea].mode())


# In[41]:

data_train.isnull().sum()


# In[42]:


# 查看类别特征
category_fea


# In[43]:
# 转化成时间格式
for data in [data_train, data_testA]:
    try:
        data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
        startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        data['issueDateDT'] = data['issueDate'].apply(lambda x: x - startdate).dt.days
    except Exception as e:
        print(f"Error occurred: {e}")
        data['issueDateDT'] = None

# In[44]:


data_train['employmentLength'].value_counts(dropna=False).sort_index()


# In[45]:


# 类别型变量转换为数值型
def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
for data in [data_train, data_testA]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)


# In[46]:


data['employmentLength'].value_counts(dropna=False).sort_index()


# In[47]:


data_train['earliesCreditLine'].value_counts()


# In[48]:


# 部分类别特征
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title', 'policyCode']
for f in cate_features:
    print(f, '类型数：', data[f].nunique())


# In[49]:


# 对等级类特征处理
for data in [data_train, data_testA]:
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})


# In[50]:


for data in [data_train, data_testA]:
    data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)


# In[51]:

# 异常值处理
def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data


# In[52]:


numerical_fea.remove('issueDate')


# In[53]:


numerical_fea.remove('issueDateDT')


# In[54]:


numerical_fea


# In[55]:


for fea in numerical_fea:
    data_train = find_outliers_by_3segama(data_train,fea)
    print(data_train[fea+'_outliers'].value_counts())
    print(data_train.groupby(fea+'_outliers')['isDefault'].sum())
    print('*'*10)


# In[56]:


# 删除异常值
for fea in numerical_fea:
    data_train = data_train[data_train[fea+'_outliers']=='正常值']
    data_train = data_train.reset_index(drop=True) 


# In[57]:


# 将连续数据离散化
# 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
data['loanAmnt_bin1'] = np.floor_divide(data['loanAmnt'], 1000)
# 通过对数函数映射到指数宽度分箱
data['loanAmnt_bin2'] = np.floor(np.log10(data['loanAmnt']))
data['loanAmnt_bin3'] = pd.qcut(data['loanAmnt'], 10, labels=False)


# In[58]:


# 特征交互
for col in ['grade', 'subGrade']: 
    temp_dict = data_train.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()

    data_train[col + '_target_mean'] = data_train[col].map(temp_dict)
    data_testA[col + '_target_mean'] = data_testA[col].map(temp_dict)


# In[59]:

# 其他衍生变量 mean 和 std
for df in [data_train, data_testA]:
    for item in ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14']:
        df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
        df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')


# In[60]:


from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


# In[61]:


# 特征编码
for col in tqdm(['employmentTitle', 'postCode', 'title','subGrade']):
    le = LabelEncoder()
    le.fit(list(data_train[col].astype(str).values) + list(data_testA[col].astype(str).values))
    data_train[col] = le.transform(list(data_train[col].astype(str).values))
    data_testA[col] = le.transform(list(data_testA[col].astype(str).values))
print('Label Encoding 完成')


# In[62]:


# 特征选择
# 删除不需要的数据
for data in [data_train, data_testA]:
    data.drop(['issueDate','id'], axis=1,inplace=True)


# In[63]:


data_train = data_train.fillna(axis=0,method='ffill')


# In[64]:
x_train = data_train.drop(['isDefault'], axis=1)
# In[65]:
x_train

# In[66]:

features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_testA[features]
y_train = data_train['isDefault']


# In[67]:


x_train.columns


# In[68]:


x_test


# In[69]:


numerical_x_train = list(x_train.select_dtypes(exclude=['object']).columns)


# In[70]:


numerical_x_train


# In[71]:


len(numerical_x_train)


# In[72]:

category_x_train = list(filter(lambda x: x not in numerical_x_train,list(x_train.columns)))
category_x_train

# In[73]:

x_train['earliesCreditLine'].value_counts


# In[74]:


x_train.drop('earliesCreditLine', axis=1, inplace=True)


# In[75]:


x_train


# In[76]:


x_test.drop('earliesCreditLine', axis=1, inplace=True)


# In[77]:


x_test


# In[78]:


y_train


# In[84]:
#LGB模型建模
from lightgbm.callback import early_stopping
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix], callbacks=[early_stopping(200)])
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                                     
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    
    return train, test, val_pred, val_y

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test,val_pred, val_y = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test,val_pred, val_y

lgb_train, lgb_test, val_pred, val_y= lgb_model(x_train, y_train, x_test)
# In[94]:
#catboost模型建模
def cv2_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
        
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    
    return train, test

# In[95]:


def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv2_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
    return cat_train, cat_test  # 添加返回语句


# In[96]:
cat_train, cat_train = cat_model(x_train, y_train, x_test)

# In[114]:
#调参
from bayes_opt import BayesianOptimization
import numpy as np

# 假设 x_train, y_train, x_test, y_test 已经准备好

# 定义贝叶斯优化的评估函数
def lgb_eval(min_child_weight, num_leaves, lambda_l2, feature_fraction, bagging_fraction, bagging_freq, learning_rate):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'min_child_weight': max(min_child_weight, 0),
        'num_leaves': int(round(num_leaves)),
        'lambda_l2': max(lambda_l2, 0),
        'feature_fraction': max(min(feature_fraction, 1), 0),
        'bagging_fraction': max(min(bagging_fraction, 1), 0),
        'bagging_freq': int(round(bagging_freq)),
        'learning_rate': max(learning_rate, 0),
        'seed': 2020,
        'n_jobs': -1,
        'silent': True,
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    cv_scores = []
    for train_index, valid_index in kf.split(x_train):
        trn_x, trn_y = x_train.iloc[train_index], y_train[train_index]
        val_x, val_y = x_train.iloc[valid_index], y_train[valid_index]

        train_matrix = lgb.Dataset(trn_x, label=trn_y)
        valid_matrix = lgb.Dataset(val_x, label=val_y)
        
        model = lgb.train(params, train_matrix, num_boost_round=10000, valid_sets=[valid_matrix], callbacks=[lgb.early_stopping(stopping_rounds=200)])
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        cv_scores.append(roc_auc_score(val_y, val_pred))
    
    return np.mean(cv_scores)

# 设置参数范围
param_bounds = {
    'min_child_weight': (0.01, 10),
    'num_leaves': (20, 40),
    'lambda_l2': (0, 5),
    'feature_fraction': (0.1, 0.9),
    'bagging_fraction': (0.8, 1),
    'bagging_freq': (1, 10),
    'learning_rate': (0.01, 0.2),
}

# 运行贝叶斯优化
optimizer = BayesianOptimization(f=lgb_eval, pbounds=param_bounds, random_state=2020)
optimizer.maximize(init_points=5, n_iter=10)

# 输出最佳参数
print("Best parameters found: ", optimizer.max['params'])
best_params = optimizer.max['params']

class ModelOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best_params = {}
        self.params = {}

    def process_params(self):
        # 获取最优参数
        self.best_params = self.optimizer.max['params']

        # 将需要整数值的参数进行取整
        self.best_params['num_leaves'] = int(round(self.best_params['num_leaves']))
        self.best_params['bagging_freq'] = int(round(self.best_params['bagging_freq']))

        # 设置固定参数
        fixed_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'n_jobs': -1,
            'seed': 2020,
            'silent': True,
        }

        # 合并参数
        self.params = {**self.best_params, **fixed_params}

    def get_params(self):
        return self.params

model_optimizer = ModelOptimizer(optimizer)
model_optimizer.process_params()
params = model_optimizer.get_params()

def cv3_model(clf, train_x, train_y, test_x, clf_name,params):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            # 使用贝叶斯优化得到的参数
            # 更新参数
            updated_params = {**params, 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': -1}
            updated_params['num_leaves'] = int(round(updated_params['num_leaves']))
            updated_params['bagging_freq'] = int(round(updated_params['bagging_freq']))

            model = clf.train(params, train_matrix, num_boost_round=50000, valid_sets=[train_matrix, valid_matrix], callbacks=[early_stopping(200)])
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            cv_scores.append(roc_auc_score(val_y, val_pred))
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                                     
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    
    return train, test, val_pred, val_y

# 使用最优参数拟合模型
lgb_train, lgb_test, val_pred, val_y = cv3_model(lgb, x_train, y_train, x_test, "lgb", params)

from sklearn.metrics import roc_curve, auc
# 验证集
fpr_val, tpr_val, _ = roc_curve(val_y, val_pred)
roc_auc_val = auc(fpr_val, tpr_val)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



