import datetime
from datetime import date
from datetime import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def about_data(data):
    print("="*60)
    print(data.info(memory_usage="deep"))
    print("+"*60)
    print("Number of samples in the dataset are ",data.shape[0])
    print("Number of features in the dataset are ",data.shape[1])
    print("Number of missing values in the dataset are ",sum(data.isnull().any()))
    print("Number of duplicate values in the dataset are ",sum(data.duplicated()))

# Some important functions
def age(born):
    x = datetime.datetime.strptime(born, '%d-%m-%y').strftime('%Y/%m/%d')
    x = pd.to_datetime(x,format="%Y/%m/%d")
    today = date.today()
    return (today.year - x.year)

def decrease(row):
    if row < 0:
        row = -1
    return row  


def col_drop(data,col):
	data.drop(columns=col,axis=1,inplace=True)


def category_to_numeric(data):
	data["Customer_occ"] = data["Customer_occ"].astype("category")
    replace_comp = {"Customer_occ":{"Salaried":0,"Self employed":1}}
    data.replace(replace_comp,inplace=True)

    labels = data["Bureau_desc"].astype("category").cat.categories.tolist()
    replace_map_comp = {'Bureau_desc' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
    data.replace(replace_map_comp,inplace=True)


def feature_engineering(data):
	data["AGE"] = data["Date.of.Birth"].apply(lambda row: age(row))
	data["PRI.CURRENT.BALANCE"] = data["PRI.CURRENT.BALANCE"].apply(lambda row:decrease(row))
    data["SEC.CURRENT.BALANCE"] = data["SEC.CURRENT.BALANCE"].apply(lambda row:decrease(row))
    data["Monthly_Average_tenure"] = data["Average_Tenure"] // 30
    data["Credit_Score"] = data["SEC.OVERDUE.ACCTS"] + data["PRI.OVERDUE.ACCTS"] + data["PRI.NO.OF.ACCTS"] + \
                         data["PRI.ACTIVE.ACCTS"] + data["SEC.NO.OF.ACCTS"] + data["SEC.ACTIVE.ACCTS"] + \
                         data["SEC.CURRENT.BALANCE"] + data["PRI.CURRENT.BALANCE"] + data["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
    data["Appraised_value"] = data["disbursed_amount"] / data["ltv"]
    data["Libaility"] = data["PRIMARY.INSTAL.AMT"] + data["SEC.INSTAL.AMT"] + data["PRI.CURRENT.BALANCE"] + data["SEC.CURRENT.BALANCE"]
    data["Asset"] = data["asset_cost"] + data["Appraised_value"]
    data["Total_debt"] = data["Libaility"] / data["Asset"]
    data["Monthly_loan"] = data["PRIMARY.INSTAL.AMT"] + data["SEC.INSTAL.AMT"]   
    data["Monthly_Income"] = (data["Asset"] - data["Libaility"]) / 12
    data["Monthly_debt"] = data["Monthly_Income"]*data["Total_debt"]
    y = data["Monthly_Average_tenure"] + 1
    data["Interest"] = ((data["Monthly_loan"]*100) / (data["disbursed_amount"]*y)) + 1
    return data


def preprocessing(data):
	col = {"Employment.Type":"Customer_occ","PERFORM_CNS.SCORE":"Bureau_Score","PERFORM_CNS.SCORE.DESCRIPTION":"Bureau_desc"}
    data.rename(columns=col,inplace=True)
	data.dropna(inplace=True)
	data = data[data["PRIMARY.INSTAL.AMT"]<3000000]
    data = data[data["SEC.INSTAL.AMT"]<1300000]
    data = data[data["PRI.CURRENT.BALANCE"]<30000000]
    data = data[data["SEC.CURRENT.BALANCE"]<3000000]
    data = data[data["PRI.SANCTIONED.AMOUNT"]<50000000]
    data = data[data["PRI.SANCTIONED.AMOUNT"]>=0]

def day_calculator(data,x,y,col,name):
	data[col] = data[col].str.replace(x," ")
    data[col] = data[col].str.replace(y," ")
    y = data[col].apply(lambda x:x.split(" "))
    final = []
    for i in range(len(data)):
    	final.append(y[i][0])
    last = []
    for i in range(len(data)):
        last.append(y[i][2])
    in_final = list(map(int, final))
    in_last = list(map(int, last))
    lst_final = [x*365 for x in in_final]
    lst_last = [x*30 for x in in_last]
    finale_list = [lst_final[i] + lst_last[i] for i in range(len(data))]
    finale_list = [x + 5 for x in finale_list]
    train[name] = pd.to_numeric(finale_list)


def data_preparation(data,col,num,seed=9):
	y = data[col].values
	data.drop(col,inplace=True,axis=1)
	X = data.values
	X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=num,random_state=seed)
	return X_train,y_train,X_test,y_test

