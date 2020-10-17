# -*- coding: utf-8 -*-

"""
Created on Fri Sep 25 09:04:28 2020
@author: Ravi Tripathi
"""

#%%
'''IMPORTING LIBRARIES'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import timedelta
from sklearn.cluster import KMeans 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
from scipy import stats

#%%
'''LOADING DATASETS AND GIVING DESCRIPTIONS'''
dfquote_s = pd.read_csv("quote-S.csv") #ticker,bid,ask,price and size wrt time
dfbar_s = pd.read_csv("bar-S.csv") #symbol,volume,acc volume,vol weighted avg 
#price,high low open close prices,avg price,epoch time at start and end 
dfbar = pd.read_csv("bar.csv") #symbol,volume,acc volume,vol weighted avg 
#price,high low open close prices,avg price,epoch time at start and end
dfevent = pd.read_csv("event.csv") #symbol,report date,system time
dfnews = pd.read_csv("news.csv") #datetime,stock and summary
dfquote = pd.read_csv("quote.csv") #ticker,bid,ask,price and size wrt time
dfrating = pd.read_csv("rating.csv") #symbol,ratingbuy,ratingoverweight,
#ratinghold,ratingunderweight,ratingsell,ratingnone,ratingscalemark,start end date
dftarget = pd.read_csv("target.csv")#symbol,updatedDate,pricetargetavg,pricetargethigh,
#pricetargetlow,numberofAnalysts

#%%
'''VARIABLES DESCRIPTION'''
#1.TICKER symbol is an arrangement of characters—usually letters—representing 
#particular securities listed on an exchange or otherwise traded publicly.
 
#2.ASK PRICE is the value point at which the seller is ready to sell. 

#3.BID PRICE is the point at which a buyer is ready to buy. 
#When the two value points match a trade takes place.

#4.ACCUMULATED VOLUME displays the total volume over an interval

#5.VWAP gives the average price a security has traded at throughout the day, based on 
#both volume and price. 

#6.EPOCH TIME is a system for describing a point in time. It is the number of seconds 
#that have elapsed since the Unix epoch

#7.BUY RATING is an investment analyst's recommendation to buy a security

#8.OVERWEIGHT RATING on a stock usually means that it deserves a higher weighting 
#than the benchmark's current weighting for that stock. 

#9.HOLD RATING is an analyst's recommendation to neither buy nor sell a security

#10.SELL RATING is a recommendation to sell a specific stock.

#11.PRICE TARGET is a price at which an analyst believes a stock to be fairly valued 
#relative to its projected and historical earnings

#%%
'''EXPLORATORY DATA ANALYSIS'''
# Let's begin with dfquote which has bid,ask price and size for tickers dated 5/8/2020
dfquote.head() #display the top 5 rows
dfquote.tail() #display the bottom 5 columns
dfquote.shape #display the rows and columns
#(2158864, 6)
dfquote.dtypes #display the data types
#time           int64
#ticker        object
#bid_price    float64
#bid_size       int64
#ask_price    float64
#ask_size       int64
dfquote.describe() #display the basic summary statistics
#              time     bid_price      bid_size     ask_price      ask_size
#count  2.158864e+06  2.158864e+06  2.158864e+06  2.158864e+06  2.158864e+06
#mean   1.616053e+01  1.658060e+02  3.824237e+00  1.678756e+02  5.382434e+00
#std    2.020090e+00  4.851936e+02  7.074522e+00  4.913153e+02  1.841143e+01
#min    1.300000e+01  3.670000e+00  1.000000e+00  3.810000e+00  1.000000e+00
#25%    1.400000e+01  2.901000e+01  1.000000e+00  2.981000e+01  1.000000e+00
#50%    1.600000e+01  5.393000e+01  1.000000e+00  5.504000e+01  1.000000e+00
#75%    1.800000e+01  8.559000e+01  4.000000e+00  8.625000e+01  4.000000e+00
#max    2.100000e+01  3.212730e+03  3.050000e+02  4.294670e+03  4.360000e+02
dfquote.isnull().sum() #check for null values
#time         0
#ticker       0
#bid_price    0
#bid_size     0
#ask_price    0
#ask_size     0
#output shows no null values
dfquote_s.head() #display the top 5 rows
dfquote_s.tail() #display the bottom 5 columns
dfquote_s.shape #display the rows and columns
#(68841, 6)
dfquote_s.dtypes #display the data types
'''time        int64
ticker        object
bid_price    float64
bid_size       int64
ask_price    float64
ask_size       int64'''
dfquote_s.describe() #display the basic summary statistics
'''            time     bid_price      bid_size     ask_price      ask_size
count  68841.000000  68841.000000  68841.000000  68841.000000  68841.000000
mean      17.517279    116.471763      6.819047    116.729414      6.673291
std        1.923703     43.893431      2.620703     44.116870      2.475730
min       12.000000     90.010000      5.000000     90.020000      5.000000
25%       16.000000     90.520000      5.000000     90.590000      5.000000
50%       19.000000     95.880000      6.000000     96.290000      6.000000
75%       19.000000    117.320000      7.000000    117.730000      7.000000
max       20.000000    596.430000     19.000000    596.980000     19.000000'''
dfquote_s.isnull().sum() #check for null values
#time         0
#ticker       0
#bid_price    0
#bid_size     0
#ask_price    0
#ask_size     0
#output shows no null values
dfquote['time'] = pd.to_datetime(dfquote['time']) #to convert the time in datetime 
#format for further analysis
dfquote['time'] = dfquote['time'].dt.hour #to change the time column to per hour so 
#that it becomes easy to analyze
dfquote.head()
dfquote_s['time'] = pd.to_datetime(dfquote_s['time']) #to convert the time in datetime 
#format for further analysis
dfquote_s['time'] = dfquote_s['time'].dt.hour #to change the time column to per hour so 
#that it becomes easy to analyze
dfquote_s.head()

#%%
'''DATA VISUALIZATION'''
sns.distplot(dfquote['bid_price']) #highly right skewed, most values b/w 0-250
sns.distplot(dfquote['bid_size']) #highly right skewed, most values b/w 0-25
sns.distplot(dfquote['ask_price']) #highly right skewed, most values b/w 0-250
sns.distplot(dfquote['ask_size']) #highly right skewed, most values b/w 0-35
#PAIR PLOT analysis with all 4 variables
cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
sns.pairplot(dfquote[cols], size = 3) #shows strong +ive corr between ask price and bid
#price
#CORRELATION COEFFICIENTS of all variable pairs
pearson_coefficeint = dfquote.corr(method='pearson')
pearson_coefficeint
'''               time  bid_price  bid_size  ask_price  ask_size
time       1.000000   0.022677  0.015829   0.021371  0.015751
bid_price  0.022677   1.000000 -0.091375   0.999788 -0.053887
bid_size   0.015829  -0.091375  1.000000  -0.091417  0.101139
ask_price  0.021371   0.999788 -0.091417   1.000000 -0.053408
ask_size   0.015751  -0.053887  0.101139  -0.053408  1.000000'''
#HEATMAP visualization of corr coeff  
sns.heatmap(pearson_coefficeint,annot=True) 
#REGPLOT for highly correlated variables
sns.regplot(x= "bid_price",y="ask_price", data = dfquote)

sns.distplot(dfquote_s['bid_price']) #highly right skewed, most values b/w 0-250
sns.distplot(dfquote_s['bid_size']) #right skewed
sns.distplot(dfquote_s['ask_price']) #highly right skewed, most values b/w 0-250
sns.distplot(dfquote_s['ask_size']) #right skewed
#PAIR PLOT analysis with all 4 variables
cols = ['bid_price', 'bid_size', 'ask_price', 'ask_size']
sns.pairplot(dfquote_s[cols], size = 3) #shows strong +ive corr between ask price and bid
#price
#CORRELATION COEFFICIENTS of all variable pairs
pearson_coefficeint = dfquote_s.corr(method='pearson')
pearson_coefficeint
'''            time  bid_price  bid_size  ask_price  ask_size
time       1.000000   0.230484  0.203901   0.227967  0.194772
bid_price  0.230484   1.000000  0.056281   0.999925  0.066524
bid_size   0.203901   0.056281  1.000000   0.056221  0.267629
ask_price  0.227967   0.999925  0.056221   1.000000  0.066459
ask_size   0.194772   0.066524  0.267629   0.066459  1.000000'''
#HEATMAP visualization of corr coeff  
sns.heatmap(pearson_coefficeint,annot=True) 
#REGPLOT for highly correlated variables
sns.regplot(x= "bid_price",y="ask_price", data = dfquote_s)

#%%
'''REGRESSION ANALYSIS'''
x = dfquote[['bid_price','bid_size']]
y = dfquote['ask_price']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3)
LR=LinearRegression()
LR.fit(x_train, y_train)
y_pred=LR.predict(x_test)
r2_score(y_test,y_pred)
#0.9994800886579274
mod1=OLS(y,x).fit()
print(mod1.summary())
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:              ask_price   R-squared (uncentered):                   1.000
Model:                            OLS   Adj. R-squared (uncentered):              1.000
Method:                 Least Squares   F-statistic:                          2.848e+09
Date:                Fri, 25 Sep 2020   Prob (F-statistic):                        0.00
Time:                        15:26:32   Log-Likelihood:                     -8.0572e+06
No. Observations:             2158864   AIC:                                  1.611e+07
Df Residuals:                 2158862   BIC:                                  1.611e+07
Df Model:                           2                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
bid_price      1.0124   1.35e-05   7.52e+04      0.000       1.012       1.012
bid_size      -0.0025      0.001     -2.961      0.003      -0.004      -0.001
==============================================================================
Omnibus:                  6518348.284   Durbin-Watson:                   1.966
Prob(Omnibus):                  0.000   Jarque-Bera (JB):   17321620184659.275
Skew:                          43.099   Prob(JB):                         0.00
Kurtosis:                   13879.465   Cond. No.                         64.0
==============================================================================

x1 = dfquote_s[['bid_price','bid_size']]
y1 = dfquote_s['ask_price']
x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1, test_size=0.3)
LR=LinearRegression()
LR.fit(x_train1, y_train1)
y_pred1=LR.predict(x_test1)
r2_score(y_test1,y_pred1)
#0.9998615212632138
mod2=OLS(y1,x1).fit()
print(mod2.summary())
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:              ask_price   R-squared (uncentered):                   1.000
Model:                            OLS   Adj. R-squared (uncentered):              1.000
Method:                 Least Squares   F-statistic:                          1.786e+09
Date:                Fri, 25 Sep 2020   Prob (F-statistic):                        0.00
Time:                        15:27:17   Log-Likelihood:                         -56258.
No. Observations:               68841   AIC:                                  1.125e+05
Df Residuals:                   68839   BIC:                                  1.125e+05
Df Model:                           2                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
bid_price      1.0037   3.54e-05   2.84e+04      0.000       1.004       1.004
bid_size      -0.0223      0.001    -37.004      0.000      -0.024      -0.021
==============================================================================
Omnibus:                   160907.722   Durbin-Watson:                   1.345
Prob(Omnibus):                  0.000   Jarque-Bera (JB):       2304336259.353
Skew:                          23.173   Prob(JB):                         0.00
Kurtosis:                     898.104   Cond. No.                         36.1
==============================================================================
#Thus, ask_price can be predicted using bid_price and bid_size

#%%
'''TIME SERIES TREND VISUALIZATION AND ANALYSIS'''
#Filtering the table based on time column 2020-08-05:
Bar_30=dfbar[dfbar['time'].str.lower().str.contains('2020-08-05')]
quote = pd.read_csv("quote.csv")
Quote_30=quote[quote['time'].str.lower().str.contains('2020-08-05')]
#Merging Two dataframes into one based on time:
Dataframe1=pd.merge(Bar_30,Quote_30,on='time')
Dataframe1.shape
'''(14842355, 17)'''
Dataframe1=Dataframe1.head(30)
Dataframe1.shape
#(30, 17)

#Visualizing bid_price, ask_price, average_price based on time:
plt.figure()
df=Dataframe1[['bid_price','ask_price','average_price']].plot(figsize=(8,8),marker='o',grid=True,markersize=5)#line plot
plt.xlabel('Time')
plt.ylabel('Price');

df1=dfrating[['symbol','ratingBuy','ratingScaleMark','consensusStartDate','consensusEndDate']]
df2=dfbar[['symbol','average_price']]
Table=pd.merge(df1,df2,on='symbol')
Table=Table[['symbol','ratingBuy','ratingScaleMark','average_price','consensusStartDate','consensusEndDate']]
Table.head()
''' symbol  ratingBuy  ...         consensusStartDate           consensusEndDate
0    ABC          9  ...  2020-08-28 00:00:00+00:00  2020-08-31 00:00:00+00:00
1    ABC          9  ...  2020-08-28 00:00:00+00:00  2020-08-31 00:00:00+00:00
2    ABC          9  ...  2020-08-28 00:00:00+00:00  2020-08-31 00:00:00+00:00
3    ABC          9  ...  2020-08-28 00:00:00+00:00  2020-08-31 00:00:00+00:00
4    ABC          9  ...  2020-08-28 00:00:00+00:00  2020-08-31 00:00:00+00:00'''
Table.shape
#(2466104, 6)

plt.figure(figsize=(30,20))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2,1,1)
sns.lineplot(data=Table, x="consensusStartDate", y="average_price",marker='o',c='#4b0082',linestyle=':')
plt.xticks(rotation=90)
plt.xlabel('consensusStartDate',fontsize=17)
plt.ylabel('average_price',fontsize=20)
plt.subplot(2,1,2)
sns.lineplot(data=Table, x='consensusEndDate',y='average_price',marker='^',c='red',markersize=8,linestyle='dashed')
plt.xticks(rotation=90)
plt.xlabel('consensusEndDate',fontsize=17)
plt.ylabel('average_price',fontsize=20)

#extracting price target average from Target data set 
target1=dftarget[dftarget['updatedDate'].str.lower().str.contains('2020-08-31')]
Target1=target1[['updatedDate','priceTargetAverage']]
#Renaming column label to perform merging 
Target2=Target1.rename(columns={'updatedDate':'reportDate'})
Target2.head()
'''     reportDate  priceTargetAverage
184  2020-08-31               59.66
185  2020-08-31               96.13
186  2020-08-31              114.13
187  2020-08-31              133.67
188  2020-08-31              183.87'''

#Merging 'priceTargetAverage' column from Target dataframe with event dataframe:
Dataframe1=pd.merge(Target2,dfevent, on='reportDate')
print(Dataframe1)
'''     reportDate  priceTargetAverage           system_time symbol
0    2020-08-31               59.66  Aug 31 18:06:44 2020    BMA
1    2020-08-31               59.66  Aug 31 18:06:44 2020    CUE
2    2020-08-31               59.66  Aug 31 18:06:44 2020   SCSC
3    2020-08-31               59.66  Aug 31 18:06:44 2020   CTLT
4    2020-08-31               59.66  Aug 31 18:06:44 2020   KRKR
..          ...                 ...                   ...    ...
295  2020-08-31               95.92  Aug 31 18:06:44 2020    BMA
296  2020-08-31               95.92  Aug 31 18:06:44 2020    CUE
297  2020-08-31               95.92  Aug 31 18:06:44 2020   SCSC
298  2020-08-31               95.92  Aug 31 18:06:44 2020   CTLT
299  2020-08-31               95.92  Aug 31 18:06:44 2020   KRKR'''

#Splitting datetime into date and time for further processing:
bar2=dfbar[dfbar['time'].str.lower().str.contains('2020-08-31')]
x=bar2['time'].str.split(" ", n = 1, expand = True)
bar2['updatedDate']= x[0] # Date
bar2['time2']=x[1] # Time
bar2.head()
'''                         time symbol  ...  updatedDate           time2
38942  2020-08-31 22:47:00+00:00   AAPL  ...   2020-08-31  22:47:00+00:00
38943  2020-08-31 22:46:00+00:00   AAPL  ...   2020-08-31  22:46:00+00:00
38944  2020-08-31 22:37:00+00:00   AAPL  ...   2020-08-31  22:37:00+00:00
38945  2020-08-31 22:36:00+00:00   AAPL  ...   2020-08-31  22:36:00+00:00
38946  2020-08-31 22:33:00+00:00    HFC  ...   2020-08-31  22:33:00+00:00'''

#looking for a common column to merge the dataframe:
bar3=bar2[['updatedDate','average_price']]
print(bar3.head(),'\n',Target1.head())
'''      updatedDate  average_price
38942  2020-08-31       129.2176
38943  2020-08-31       129.2176
38944  2020-08-31       129.2176
38945  2020-08-31       129.2176
38946  2020-08-31        24.1322 
     updatedDate  priceTargetAverage
184  2020-08-31               59.66
185  2020-08-31               96.13
186  2020-08-31              114.13
187  2020-08-31              133.67
188  2020-08-31              183.87'''

#Merging dataframes by updatedDate
Dataframe2=pd.merge(bar3,Target1,on='updatedDate')
Dataframe2.drop('priceTargetAverage',axis=1,inplace=True)
print(Dataframe2)
'''     updatedDate  average_price
0       2020-08-31       129.2176
1       2020-08-31       129.2176
2       2020-08-31       129.2176
3       2020-08-31       129.2176
4       2020-08-31       129.2176
           ...            ...
533935  2020-08-31        50.8081
533936  2020-08-31        50.8081
533937  2020-08-31        50.8081
533938  2020-08-31        50.8081
533939  2020-08-31        50.8081'''

Dataframe2.plot(figsize=(18,5), grid=True)
plt.xlabel('Time', fontsize=15)
plt.ylabel('average_price', fontsize=15)

#to find the trend in price prior and post updated date
bar5=bar2
bar5=bar5.rename(columns={'updatedDate':'date','time':'datetime','time2':'time'})
bar5.columns

# Selecting columns to merge from preprocessed bar dataframe: 
bar5=bar5[['datetime','average_price','time']]

#Merging dataframes for final Dataframe:
Dataframe3=pd.merge(dfnews,bar5,on='datetime')
Dataframe3
'''                      datetime stock  ... average_price            time
0    2020-08-31 22:09:00+00:00  ZION  ...      129.2176  22:09:00+00:00
1    2020-08-31 21:21:00+00:00  TSCO  ...       84.6665  21:21:00+00:00
2    2020-08-31 20:53:00+00:00   CRM  ...        6.4145  20:53:00+00:00
3    2020-08-31 20:45:00+00:00   DIS  ...        6.4145  20:45:00+00:00
4    2020-08-31 20:34:00+00:00   MMC  ...        6.4145  20:34:00+00:00
..                         ...   ...  ...           ...             ...
527  2020-08-31 15:16:00+00:00  TWTR  ...       49.9126  15:16:00+00:00
528  2020-08-31 15:16:00+00:00  TWTR  ...       69.0889  15:16:00+00:00
529  2020-08-31 15:16:00+00:00  TWTR  ...      203.2878  15:16:00+00:00
530  2020-08-31 15:16:00+00:00  TWTR  ...       63.8583  15:16:00+00:00
531  2020-08-31 15:16:00+00:00  TWTR  ...       50.8081  15:16:00+00:00'''

plt.figure()
Dataframe3.plot(figsize=(15,5),c='purple')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Average Price',fontsize=12)
plt.title('Average Price Distribution', fontsize=15)

#%%
'''KMEANS CLUSTERING BASED ON PRICE PARAMETERS'''
bar_S = pd.read_csv('bar-S.csv')
quote = pd.read_csv('quote-S.csv')
# CREATE CLUSTER N1 using open, close, high, low, average and VWAP prices to group the stocks into different clusters
#calculating the mean of these prices in a 5-hour window of the day and then performing the clustering

bar_S = bar_S.sort_values(by='time')
bar_S['date'] = pd.to_datetime(bar_S['time']).apply(lambda x: x.strftime('%Y-%m-%d'))
#analysis for 30 windows of 10 mins of each day leading to 5 hrs of stock movement 

stock_list = list(set(bar_S['symbol']))
df_list = []
df_list_2 = []
window = 30
n_minutes = 10
date_list = list(set(bar_S['date']))
features = ['open_price','high_price','low_price','close_price',
            'average_price','VWAP','volume','accumulated_volume','symbol','date','time']
bar_S_window_time = 0
for i in stock_list:
    new_bar_S = bar_S[bar_S['symbol']==i]
    for j in date_list: 
        try:
            lower_limit = list(new_bar_S[new_bar_S['date']==j].sort_values(by='time')['time'])[0]
            lower_limit = pd.to_datetime(lower_limit)
            upper_limit = lower_limit + timedelta(minutes=window*n_minutes)
            df = new_bar_S[(new_bar_S['date']==j) & (new_bar_S['time']>=str(lower_limit)) & (new_bar_S['time']<str(upper_limit))]
            sorted_df = df.sort_values(by='time')[features]
            df_list_2.append(sorted_df)
            sorted_df = sorted_df.groupby('date').mean()
            sorted_df['symbol'] = i
            df_list.append(sorted_df)
        except Exception as e:
            pass
bar_S_windowed = pd.concat(df_list)
bar_S_windowed = bar_S_windowed.sort_index()
bar_S_window_time = pd.concat(df_list_2)
bar_S_col_list = list(bar_S_windowed)
bar_S_windowed.columns = ['mean '+col if col!='symbol' else 'symbol' for col in bar_S_col_list]
bar_S_windowed.head()
'''            mean open_price  mean high_price  ...  mean accumulated_volume  symbol
date                                          ...                                 
2020-08-03       533.330000       533.345750  ...             12509.450000     BIO
2020-08-03       179.882843       179.895539  ...             19028.970588    IPGP
2020-08-03       101.457267       101.462438  ...             68571.506849     BBY
2020-08-03       374.672083       374.682917  ...             22339.222222     TFX
2020-08-03       187.950469       187.959180  ...             34356.109375    VRSK'''

#ELBOW CURVE FOR DETERMINING k IN KMEANS CLUSTERING

price_features = ['mean open_price','mean high_price','mean low_price','mean close_price',
                  'mean average_price','mean VWAP']
count = 0
cm = plt.get_cmap('rainbow')
colors = cm(np.linspace(0, 1, 24))
plt.rcParams['figure.figsize'] = (10,10)
for index in date_list: 
    bar_S_price_data = bar_S_windowed.loc[index,price_features]
    cluster_list = range(1,21)
    wss_list = []
    for i in cluster_list:
        km = KMeans(n_clusters=i)
        km.fit(bar_S_price_data)
        wss_list.append(km.inertia_)
    plt.plot(cluster_list,wss_list,marker='o',color=colors[count])
    plt.xlabel('No. of clusters')
    plt.ylabel('Within sum of squares (WSS)')
    plt.title('Elbow Curve')
    plt.legend(date_list)
    count+=1
    for i, txt in enumerate(cluster_list):
        plt.annotate(txt, (cluster_list[i], wss_list[i]))
        
# For majority of dates, the elbow point appears to be 4 as no. of clusters. 
# So, no.of clusters can be taken as 4.

#Performing KMEANS
km = KMeans(n_clusters=4)
bar_S_price_data =  bar_S_windowed.loc[:,price_features]
km.fit(bar_S_price_data)
label = km.predict(bar_S_price_data)

#Listing out stocks in each cluster
stock_cluster = dict(zip(bar_S_windowed.loc[:,'symbol'],label))
cluster_0_N1=[]
cluster_1_N1=[]
cluster_2_N1= []
cluster_3_N1=[]
for key in stock_cluster:
    if stock_cluster[key]==0:
        cluster_0_N1.append(key)    
    elif stock_cluster[key]==1:
        cluster_1_N1.append(key)
    elif stock_cluster[key]==2:
        cluster_2_N1.append(key)
    else:
        cluster_3_N1.append(key)
print('Stocks belonging to cluster 0:',','.join(set(cluster_0_N1)))
print('\nStocks belonging to cluster 1:',','.join(set(cluster_1_N1)))
print('\nStocks belonging to cluster 2:',','.join(set(cluster_2_N1)))
print('\nStocks belonging to cluster 3:',','.join(set(cluster_3_N1)))

'''Stocks belonging to cluster 0: 
ABT,GRMN,ATO,AMD,BDX,GPN,IT,EL,TT,ZBH,RE,ITW,CLX,ADP,ETR,ANET,STE,V,ALL,SNPS,XLNX,AXP,
GPC,JKHY,WEC,CI,RMD,DTE,WYNN,ODFL,TTWO,HON,ADI,MMM,TSCO,EXR,QCOM,ULTA,APTV,CPRT,PNC,HLT,DRI,HD,WAT,PLD,TROW,CTXS,EFX,PPG,
SYK,MCHP,KMB,FISV,IFF,SNA,PH,TRV,MLM,CDW,CDNS,DIS,CHD,HSY,STZ,ACN,URI,PKG,HCA,FFIV,PEP,EA,QRVO,CBOE,RSG,LLY,CAT,ETN,SIVB,
TXN,DGX,CMI,J,IEX,TMUS,GS,AMGN,PAYC,DLR,ROST,ABMD,AAP,KLAC,TGT,NSC,SJM,UHS,INCY,AIZ,MAR,ANTM,CE,BA,MDT,TIF,IBM,BRK.B,ALB,
APH,BIIB,AME,CVX,ZTS,AWK,A,PKI,EXPE,VRSN,PCAR,CCI,ES,LDOS,BR,JNJ,LOW,MMC,DLTR,FIS,KMX,CRM,WHR,MSI,LH,YUM,ADSK,PYPL,MTB,
TEL,AVY,CB,FTNT,MCD,MAA,IPGP,IQV,VRSK,ROK,FDX,JBHT,DG,PXD,FB,KEYS,SWK,EXPD,SRE,MHK,AAPL,WM,FLT,ABC,AON,JPM,MKC,AMP,ARE,
ZBRA,VMC,WST,ALLE,MSFT,SWKS,PG,CHRW,ESS,ECL,AMT,CME,PGR,ICE,ALXN,ABBV,UNP,BXP,HII,MCK,BBY,UPS,AVB,AJG,AKAM,NDAQ,WMT,KSU,
LIN,VRTX,FMC,FRC,DE,GD,WLTW,LHX,NKE,PSA,VAR,DOV,DHR

Stocks belonging to cluster 1: NVR,AMZN

Stocks belonging to cluster 2: MTD,AZO,CMG,GOOGL,GOOG,BKNG

Stocks belonging to cluster 3: IDXX,BIO,GWW,MA,ISRG,CTAS,TMO,TYL,EQIX,MSCI,SBAC,MCO,TDY,ADBE,BLK,DPZ,COO,CHTR,APD,COST,ALGN,NEE,LMT,TFX,MKTX,DXCM,AVGO,ILMN,SPGI,ANSS,HUM,ORLY,UNH,SHW,NOW,LRCX,NOC,NVDA,NFLX,INTU,REGN,TDG,ROP'''

'''Displaying MOVEMENT OF AVERAGE DAILY PRICES of 4 stocks in each cluster'''
 
def plot_price_curve(cluster,n_stocks):
    plt.rcParams['figure.figsize']=(50,20)
    count = 1
    for stock in cluster[:n_stocks]:
        df = bar_S_windowed[bar_S_windowed['symbol']==stock]
        features = ['mean open_price','mean high_price','mean low_price','mean close_price',
                    'mean average_price','mean VWAP','mean volume','mean accumulated_volume']
        for feature in features:
            plt.title('For stock '+stock)
            plt.subplot(n_stocks,8,count)
            sns.lineplot(data = df,x = df.index,y = feature)
            count+=1 
            
print('For cluster 0 of N1')
plot_price_curve(cluster_0_N1,4)

print('For cluster 1 of N1')
plot_price_curve(cluster_1_N1,4)

print('For cluster 2 of N1')
plot_price_curve(cluster_2_N1,4)

print('For cluster 3 of N1')
plot_price_curve(cluster_3_N1,4)

#%%
'''KMEANS CLUSTERING BASED ON RETURNS'''
#creating N2 cluster based on returns which are of two types: bidprice returns and ask returns, 
#taken into consideration while grouping. Also, variation of cumulative bid price 
#with respect to the stock volume.

quote = quote.sort_values(by='time')
quote['date'] = pd.to_datetime(quote['time']).apply(lambda x: x.strftime('%Y-%m-%d'))

stock_list = list(set(quote['ticker']))
df_list = []
for i in stock_list:
    df = quote[quote['ticker']==i]
    df['bid_price_change'] = df['bid_price'].diff().values
    df['bid_price_returns'] = df['bid_price_change']/df['bid_price']
    df['ask_price_returns'] = df['ask_price'].diff().values/df['ask_price'].values
    average_bid_price = df['bid_price'].median()
    df['bid_price_volatility'] = (((df['bid_price'] - average_bid_price)**2)/len(df))**0.5 
    df_list.append(df)
quote = pd.concat(df_list)

df_list = []
window = 30
n_minutes = 10
date_list = list(set(quote['date']))
for i in stock_list:
    new_quote_S = quote[quote['ticker']==i]
    for j in date_list: 
        try:
            lower_limit = list(new_quote_S[new_quote_S['date']==j].sort_values(by='time')['time'])[0]
            lower_limit = pd.to_datetime(lower_limit)
            upper_limit = lower_limit + timedelta(minutes=window*n_minutes)
            df = new_quote_S[(new_quote_S['date']==j) & (new_quote_S['time']>=str(lower_limit)) & 
                             (new_quote_S['time']<str(upper_limit))]
            #Filling null values in records with 0 as default
            df['bid_price_change'] = df['bid_price_change'].fillna(0)
            df['bid_price_returns'] = df['bid_price_returns'].fillna(0)
            df['ask_price_returns'] = df['ask_price_returns'].fillna(0)
            df = df.sort_values(by='time')
            df['cumulative bid_price'] = np.cumsum(df['bid_price'])
            df_list.append(df)
        except Exception as e:
            pass
quote_S_windowed = pd.concat(df_list)
quote_S_windowed = quote_S_windowed.sort_index()
quote_S_windowed.index = quote_S_windowed['date']
quote_S_windowed.drop('date',axis=1,inplace=True)
quote_S_windowed.head()
'''                                 time  ... cumulative bid_price
date                                   ...                     
2020-09-11  2020-09-11 19:59:10+00:00  ...               100.00
2020-09-11  2020-09-11 19:46:55+00:00  ...               228.03
2020-09-11  2020-09-11 19:45:09+00:00  ...               117.76
2020-09-11  2020-09-11 19:44:17+00:00  ...                93.80
2020-09-11  2020-09-11 19:42:47+00:00  ...               477.24'''

#Cumulative bid price vs Volume
#Merging quote-S and bar-S data for further analysis

bar_quote_data = quote_S_windowed.merge(bar_S,left_on=['ticker','time'],right_on=['symbol','time'],how='inner')
plt.title('cumulative bid price vs volume bubble chart based on ticker-wise bid size')
sns.scatterplot(data=bar_quote_data,x='volume',y='cumulative bid_price',hue='ticker',size='bid_size')

#ELBOW CURVE FOR DETERMINING k IN KMEANS CLUSTERING
def plot_elbow(features):
    cm = plt.get_cmap('rainbow')
    date_list = list(set(quote_S_windowed.index))
    colors = cm(np.linspace(0, 1, len(date_list)))
    count=0
    plt.rcParams['figure.figsize'] = (10,10)
    for index in date_list: 
        quote_S_data = quote_S_windowed.loc[index,features]
        cluster_list = range(1,21)
        wss_list = []
        for i in cluster_list:
            km = KMeans(n_clusters=i)
            km.fit(quote_S_data)
            wss_list.append(km.inertia_)
        plt.plot(cluster_list,wss_list,marker='o',color=colors[count])
        plt.xlabel('No. of clusters')
        plt.ylabel('Within sum of squares (WSS)')
        plt.title('Elbow Curve')
        plt.legend(date_list)
        count+=1
        for i, txt in enumerate(cluster_list):
            plt.annotate(txt, (cluster_list[i], wss_list[i]))
            
returns_features = ['bid_price_returns','ask_price_returns']
plot_elbow(returns_features)
# the elbow point of optimum no.of clusters appears to be 2 for majority of the dates. 
#So, N2 value, which is based on bid price return and ask price returns is 2.

km = KMeans(n_clusters=2)
quote_S_returns_data = quote_S_windowed.loc[:,returns_features]
km.fit(quote_S_returns_data)
label = km.predict(quote_S_returns_data)

'''Listing out stocks based on clusters in N2'''
stock_cluster = dict(zip(quote_S_windowed.loc[:,'ticker'],label))
cluster_0_N2=[]
cluster_1_N2=[]
for key in stock_cluster:
    if stock_cluster[key]==0:
        cluster_0_N2.append(key)    
    else:
        cluster_1_N2.append(key)
print('Stocks belonging to cluster 0:',','.join(set(cluster_0_N2)))
print('\nStocks belonging to cluster 1:',','.join(set(cluster_1_N2)))

'''Stocks belonging to cluster 0: ABT,GRMN,ATO,BDX,GPN,EL,CLX,ADP,ETR,ANET,V,ALL,XLNX,AXP,JKHY,COST,CI,
DTE,TTWO,ADI,TSCO,MMM,QCOM,EXR,ULTA,APTV,CPRT,PNC,DRI,HLT,HD,PLD,CTXS,EFX,PPG,SYK,KMB,FISV,PH,TRV,CDW,
CDNS,DIS,CHD,STZ,ACN,HCA,PEP,EA,QRVO,CBOE,CAT,ALGN,ETN,ROST,DGX,CMI,GS,DLR,AMGN,AAP,KLAC,TGT,NSC,ANTM,
CE,BA,MDT,BRK.B,ALB,APH,BIIB,AME,CVX,AWK,A,PKI,EXPE,CCI,ES,LDOS,BR,LOW,MMC,DLTR,FIS,CRM,KMX,MSI,YUM,PYPL,
ADSK,AVGO,TEL,AVY,CB,FTNT,MCD,IQV,FDX,DG,PXD,FB,KEYS,SRE,AAPL,FLT,AON,ABC,JPM,AMP,ARE,ALLE,SWKS,PG,CHRW,
CME,ECL,AMT,ESS,APD,DPZ,ICE,ABBV,ALXN,CHTR,BXP,UPS,BBY,NEE,AVB,AJG,AKAM,LIN,WMT,KSU,FMC,DE,GD,LHX,NKE,TDG,
DOV,DHR

Stocks belonging to cluster 1:
'''
#Thus, all points (homogeneous) lie in a single cluster

#%%
'''KMEANS CLUSTERING BASED ON BID SIZE'''

'''BID PRICE VOLATILITY, BID PRICE CHANGE AND EXPLORING TRENDS OVER TIME'''

#clustering stocks based on bid size and quote data.
#also calculating bid price volatility and bid price change and exploring its trends over time in 
#these clusters.

bid_size_feature = ['bid_size']
plot_elbow(bid_size_feature)

#So, the elbow point of optimum no.of clusters appears to be 3 for majority of the dates. 
#So, N3 value, which is based on bid size is 3.

km = KMeans(n_clusters=3)
quote_S_size_data = quote_S_windowed.loc[:,bid_size_feature]
km.fit(quote_S_size_data)
label = km.predict(quote_S_size_data)

'''Listing out stocks based on N3'''
stock_cluster = dict(zip(quote_S_windowed.loc[:,'ticker'],label))
cluster_0_N3=[]
cluster_1_N3=[]
cluster_2_N3=[]
for key in stock_cluster:
    if stock_cluster[key]==0:
        cluster_0_N3.append(key)    
    elif stock_cluster[key]==1:
        cluster_1_N3.append(key)
    else:
        cluster_2_N3.append(key)
cluster_0_N3 = list(set(cluster_0_N3))
cluster_1_N3 = list(set(cluster_1_N3))
cluster_2_N3 = list(set(cluster_2_N3))
print('Stocks belonging to cluster 0:',','.join(cluster_0_N3))
print('\nStocks belonging to cluster 1:',','.join(cluster_1_N3))
print('\nStocks belonging to cluster 2:',','.join(cluster_2_N3))

'''Stocks belonging to cluster 0: ABT,GRMN,BDX,GPN,EL,CLX,ETR,V,ALL,COST,AXP,JKHY,CI,DTE,TTWO,ADI,TSCO,
MMM,QCOM,EXR,ULTA,APTV,HLT,PNC,DRI,HD,PLD,CTXS,EFX,PPG,SYK,KMB,FISV,PH,CDNS,DIS,CHD,STZ,ACN,HCA,PEP,EA,
QRVO,CBOE,CAT,ETN,ROST,DGX,CMI,GS,DLR,AAP,KLAC,TGT,NSC,ANTM,MDT,BRK.B,ALB,APH,BIIB,AME,CVX,AWK,A,PKI,CCI,
ES,LDOS,BR,LOW,MMC,DLTR,CRM,KMX,MSI,YUM,PYPL,ADSK,AVGO,TEL,AVY,CB,FTNT,IQV,DG,FB,KEYS,SRE,AAPL,FLT,AMP,
ARE,ALLE,SWKS,PG,CHRW,CME,ECL,AMT,DPZ,CHTR,ICE,ALXN,BXP,UPS,NEE,AVB,AJG,AKAM,LIN,WMT,FMC,DE,GD,LHX,NKE,
TDG,DOV,DHR

Stocks belonging to cluster 1: CPRT,BA,APD,PXD,ALGN

Stocks belonging to cluster 2: ATO,TRV,CDW,ABC,AON,JPM,ADP,EXPE,ANET,ESS,ABBV,XLNX,FIS,BBY,AMGN,KSU,MCD,
CE,FDX'''

def plot_bid_curve(cluster,n_stocks):
    count = 1
    cluster = list(set(cluster))
    plt.rcParams['figure.figsize'] = (20,20)
    for stock in cluster[:n_stocks]: 
        df = quote_S_windowed[['time','bid_price_change','ticker','bid_size','bid_price_volatility','cumulative bid_price']]
        df = df[df['ticker']==stock].sort_values(by='time')
        plt.subplot(n_stocks,4,count)
        plt.plot(list(df['bid_price_change']))
        plt.xlabel('time')
        plt.ylabel('bid price change')
        plt.title('For stock '+stock)
        count+=1
        plt.subplot(n_stocks,4,count)
        plt.plot(list(df['bid_size']))
        plt.xlabel('time')
        plt.ylabel('bid size')
        plt.title('For stock '+stock)
        count+=1
        plt.subplot(n_stocks,4,count)
        plt.plot(list(df['bid_price_volatility']))
        plt.xlabel('time')
        plt.ylabel('bid price volatility')
        plt.title('For stock '+stock)
        count+=1
        plt.subplot(n_stocks,4,count)
        plt.plot(list(df['cumulative bid_price']))
        plt.xlabel('time')
        plt.ylabel('cumulative bid price')
        plt.title('For stock '+stock)
        count+=1

print('Cluster 0 bid plot:')
plot_bid_curve(cluster_0_N3,4)

print('Cluster 1 of N3 bid plot:')
plot_bid_curve(cluster_1_N3,4)

print('Cluster 2 of N3 bid plot:')
plot_bid_curve(cluster_2_N3,4)

#%%
'''OPTIMIZING N1 BY ARIMA MODEL ACCURACY'''
'''PREDICTIONS USING ARIMA'''

#running ARIMA model is computationally expensive and also displaying predictions on graph for each stock 
#of cluster N1 is cumbersome, hence running for only one stock from each clusters of N1.

def stationary_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dftest[2]

def ARIMA_prediction_plot(timeseries,split):
    train_ts = timeseries[:split]
    test_ts = timeseries[split:]
    p_val = stationary_test(timeseries)
    model = ARIMA(train_ts, order=(p_val,1,2))  
    results_ARIMA = model.fit(disp=-1)
    #Future Forecasting
    history = list(train_ts)
    predictions = []
    test_data = list(test_ts)
    for i in range(len(test_data)):
        model = ARIMA(history, order=(p_val,1,2))
        model_fit = model.fit(disp=-1)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(float(yhat))
        history.append(float(yhat))
    plt.rcParams['figure.figsize'] = (20,10)
    plt.plot(timeseries)
    plt.plot(test_ts.index,predictions,color='green')
    plt.axvline(train_ts.index[-1],color='orange',dashes=(5,2,1,2))
    plt.xlabel('Average price')
    plt.ylabel('time')
    plt.legend(['actual values','predicted values for test data'])
    
bar_S_window_time.index = bar_S_window_time['time']
bar_S_window_time = bar_S_window_time.sort_index()

stock = cluster_0_N1[0]
print('For N1 cluster 0 stock: ',stock)
'''For N1 cluster 0 stock:  IPGP'''
timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']
ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))

stock = cluster_1_N1[0]
print('For N1 cluster 1 stock: ',stock)
'''For N1 cluster 1 stock:  NVR'''
timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']
ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))

stock = cluster_2_N1[0]
print('For N1 cluster 2 stock: ',stock)
'''For N1 cluster 2 stock:  CMG'''
timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']
ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))

stock = cluster_3_N1[0]
print('For N1 cluster 3 stock: ',stock)
'''For N1 cluster 3 stock:  BIO'''
timeseries = bar_S_window_time[bar_S_window_time['symbol']==stock]['average_price']
ARIMA_prediction_plot(timeseries,int(0.90*len(timeseries)))
#%%