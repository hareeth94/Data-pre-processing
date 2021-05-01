import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import cx_Oracle
import xlrd
import xlsxwriter
from io import BytesIO

temp = "//temp.csv"
path = os.getcwd()
path = path + temp
global df

def upload_file():
    st.sidebar.header("Data Import")
    
    f_option = (".xlsx",".csv","oracle")
    f_select = st.sidebar.radio("Please select file type",f_option)
    
    if f_select == ".xlsx":
        df = xl()
        return df
        
    elif f_select == ".csv":
        df = csv()
        return df
        
    elif f_select == "oracle":
        st.info("Enter Oracle database information")
        
        user = st.text_input("Enter user name")
        password = st.text_input("Enter password")
        host = st.text_input("Enter host address")
        port = st.text_input("Enter port number")
        query = st.text_input("Enter  query for data")
        
        if st.button("Connect"):
            con_query = "{}/{}@{}:{}/ORCL".format(user,password,host,port)
            
            con = cx_oracle.connect(con_query)
            
            if con!=None:
                st.info("Connection established successfully")
                df = pd.read_sql(query,con)
                st.dataframe(df)
                df.to_csv(path)
                return df
                
                
    
def xl():
    fl_up = st.sidebar.file_uploader("Choose a file", type = "xlsx")
    if st.sidebar.button("Upload File"):
        df = pd.read_excel(fl_up)
        st.dataframe(df)
        df.to_csv(path)
        return df

def csv():
    fl_up = st.sidebar.file_uploader("Choose a file", type = "csv")
    if st.sidebar.button("Upload File"):
        df = pd.read_csv(fl_up)
        st.dataframe(df)
        st.write(df)
        df.to_csv(path)
        return df
    
def export(df):
    st.sidebar.header("Data Export")
    
    d_option = (".xlsx",".csv")
    d_select = st.sidebar.radio("Please select file type",d_option)
    
    if d_select == ".xlsx":
        if st.sidebar.button("Download File"):
            st.sidebar.markdown(get_download_link_excel(df), unsafe_allow_html = True)
        
    elif d_select == ".csv":
        if st.sidebar.button("Download CSV"):
            st.sidebar.markdown(get_download_link_csv(df), unsafe_allow_html = True)
            
    elif d_select == "oracle":
        st.info("Enter Oracle database information")
        
        user = st.text_input("Enter user name")
        password = st.text_input("Enter password")
        host = st.text_input("Enter host address")
        port = st.text_input("Enter port number")
        table = st.text_input("Enter  table name")
        
        if st.button("Connect Oracle"):
            df = pd.read_csv(path)
            con = create_engine('oracle+cx_oracle://{}/{}@{}:{}/ORCL'.format(user,password,host,port))
            dt.to_sql('{}'.format(table), con , if_exists = 'replace')
            if con == None:
                st.info("Connection established successfully")
                
        

def get_download_link_csv(df):
    
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    
    return f'<a href="data:file/csv;base64,{b64}" download="myfile.csv">Download CSV file</a>'

def get_download_link_excel(df):
    
    val = to_excel(df)
    b64 = base64.b64encode(val)
    
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="myfiles.xlsx">Download Xlsx file</a>'


def to_excel(df):
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df.to_excel(writer)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

    
def MVMean(f,g,x,col):
    st.text("Mean Imputer")
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            imputer = imp.fit(x[col_sel])
            x[col_sel] = imputer.transform(x[col_sel])
            
        except:
            st.info("Select atlest one column")
        
    else:
        imputer = imp.fit(x)
        x = imputer.transform(x)
    return x

def MVMedian(f,g,x,col):
    st.text("Median Imputer")

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values = np.nan, strategy = 'median')
    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            imputer = imp.fit(x[col_sel])
            x[col_sel] = imputer.transform(x[col_sel])
        except:
            st.info("Select atlest one column")
        
    else:
        imputer = imp.fit(x)
        x = imputer.transform(x)
        
        
    return x

def MVMode(f,g,x,col):
    st.text("Mode Imputer")

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            imputer = imp.fit(x[col_sel])
            x[col_sel] = imputer.transform(x[col_sel])

        except:
            st.info("Select atlest one column")
        
        
    else:
        imputer = imp.fit(x)
        x = imputer.transform(x)
   
    return x

def MVKNN(f,g,x,col):
    st.text("KNN Imputer")

    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors = 2)
    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            imputer = imp.fit(x[col_sel])
            x[col_sel] = imputer.transform(x[col_sel])
        except:
            st.info("Select atlest one column")
        
    else:
        imputer = imp.fit(x)
        x = imputer.transform(x)
    return x

def outlier(x):

    col = x.columns
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    print(q1,q3)
    iqr = q3 - q1
    x = x[~((x < (q1 - 1.5*iqr)) | (x > (q3 + 1.5*iqr))).any(axis=1)]
    return x

def stdscaler(f,g,x,col):
    x_std = x
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    col = x_std.columns
    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            x_std[col_sel] = sc_x.fit_transform(x_std[col_sel])
        except:
            st.info("Select atlest one column")
        
    else:

        x_std = sc_x.fit_transform(x_std)
    return x_std
    
#Min Max Scaler
def MinmaxScaler(f,g,x,col):
    x_mm = x
    from sklearn.preprocessing import MinMaxScaler
    mm_x = MinMaxScaler()

    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            x_mm[col_sel] = mm_x.fit_transform(x_mm[col_sel])
        except:
            st.info("Select atlest one column")
        
        
    else:

        x_mm = mm_x.fit_transform(x_mm)
    return x_mm
    
#Robust
def Robust(f,g,x,col):
    x_rb = x
    from sklearn.preprocessing import RobustScaler
    rb_x = RobustScaler()

    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            x_rb[col_sel] = rb_x.fit_transform(x_rb[col_sel])
        except:
            st.info("Select atlest one column")
        
    else:

        x_rb = rb_x.fit_transform(x_rb)
    return x_rb
    
#Max Absolute
def MaxAbsolute(f,g,x,col):
    from sklearn.preprocessing import MaxAbsScaler
    ma_x = MaxAbsScaler()
    x_ma = x

    ch6 = st.radio("Do you want to slice the table:", ("Yes","No"), key = f)
    if ch6 == "Yes":
        try:
            col_sel = st.multiselect("Please select the columns",col, key = g)
            x_ma[col_sel] = ma_x.fit_transform(x_ma[col_sel])
        except:
            st.info("Select atlest one column")
        
    else:

        x_ma = ma_x.fit_transform(x_ma)
    return x_ma

def missing_value_treatment(b,f,g,df,col):
    ch2 = st.selectbox("Select an option", ["Mean", "Median", "Mode", "KNN"], key = b)
    st.write("You selected", ch2)
    if ch2 == "Mean":
        df = MVMean(f,g,df,col)
        df = pd.DataFrame(df) 

    elif ch2 == "Median":
        df = MVMedian(f,g,df,col)
        df = pd.DataFrame(df) 

    elif ch2 == "Mode":
        df = MVMode(f,g,df,col)
        df = pd.DataFrame(df) 

    elif ch2 == "KNN":
        df = MVKNN(f,g,df,col)
        df = pd.DataFrame(df) 
    
    return df
 
def feature_scaling(c,f,g,df,col):
    ch3 = st.selectbox("Choose an option", ("Std scaler", "Minmax scaler", "Robust scaler", "Maxabs scaler"), key = c)
    st.write("You selected", ch3)
    if ch3 == "Std scaler":
        df = stdscaler(f,g,df,col)

    elif ch3 == "Minmax scaler":
        df = MinmaxScaler(f,g,df,col)

    elif ch3 == "Robust scaler":
        df = Robust(f,g,df,col)

    elif ch3 == "Maxabs scaler":
        df = MaxAbsolute(f,g,df,col)
    return df
       

def main():
    
    st.title("DATA PRE-PROCESSING")
    upload_file()
    a = 1
    b = 101
    c = 201
    d = 301
    e = 401
    f = 501
    g = 601
    n = 0
    df = pd.read_csv(path)
    column = df.columns
    while n<3:
        ch1 = st.radio("Choose an option", ("Missing value", "Outlier", "Feature scaling"), key = a)
        st.write("You selected", ch1)
        
        if ch1 == "Missing value":
            df = missing_value_treatment(b,f,g,df,column) 
            st.write(df)
            n = n + 1
            st.write(n)
    
        elif ch1 =="Outlier":

            df = outlier(df)
            df = pd.DataFrame(df)
            st.write(df)
            n = n + 1
            st.write(n)
            
        else:

            df = feature_scaling(c,f,g,df,column)
            df = pd.DataFrame(df)
            st.write(df)
            n = n + 1
            st.write(n)
            
        ch4 = st.radio("Do you want to continue:", ("Yes","No"),index = 1, key = d)
        if ch4 == "Yes":
            if n < 3:
                a = a + 1
                b = b + 1
                c = c + 1
                d = d + 1
                e = e + 1
                f = f + 1
                g = g + 1
                
            else:
                st.success("Successful all 3 process completed")
                
        elif (ch4 == "No" and n<3):
            st.warning("Warning, all 3 process not completed")
            ch5 = st.radio("Do you want to continue:", ("Yes","No"),index = 1, key = e)
            if ch5 == "Yes":
                a = a + 1
                b = b + 1
                c = c + 1
                d = d + 1
                e = e + 1
                f = f + 1
                g = g + 1
            else:
                break
    export(df)
 

main()