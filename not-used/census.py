import pandas as pd
import json

def extract(path,dest,info_type):
    df=pd.read_csv(path)
    df=df.iloc[4646796:7235340]
    df=df[df['Member ID: Profile of Census Tracts (2247)'].isin(info_type)]
    df.to_csv(dest)
    return

def read_population(csv_path, CT_mapping_path, out):
    with open(CT_mapping_path, 'r') as f:
        D = json.load(f)
    CTs = D['CTNAME']
    CTs = ['535'+x for x in CTs]
    CTs = set(CTs)
    print("need CTs:",len(CTs))
    df=pd.read_csv(csv_path)
    df = df[df['Member ID: Profile of Census Tracts (2247)']==1]
    df = df[df['GEO_CODE (POR)'].isin(CTs)]
    print("got CTs:", len(df))
    df=df[['GEO_CODE (POR)','Dim: Sex (3): Member ID: [1]: Total - Sex']]
    df.columns=['code','population']
    df.to_csv(out)
    return

def assign_avg(csv_path, CT_mapping_path, out):
    # load mapping
    with open(CT_mapping_path, 'r') as f:
        D = json.load(f)
    CTs = D['CTNAME']
    x_p = D['x_p']
    y_p = D['y_p']
    roadID = D['roadID']

    #dict = {'CT_code': CTs, 'degree': x_p, 'score': y_p, 'roadID': roadID}
    # only take start point
    dict = {'CT_code': CTs[::2], 'degree': x_p[::2], 'score': y_p[::2], 'roadID': roadID[::2]}


    df_mapping = pd.DataFrame(dict)
    #count=df_mapping.groupby(['CT_code']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    count = df_mapping.groupby(['CT_code']).size().reset_index(name='counts').sort_values(by='CT_code')
    df2 = pd.read_csv(csv_path)
    df2['code']=df2['code'].map(lambda  x: format(x,'.2f')[3:])
    df2 = df2.sort_values(by='code')
    count['avg']=df2['population']/count['counts']
    count.to_csv(out)
    return





if __name__ == "__main__":
    # info=[1,51,52,53,54,55,56,57,58]
    # source='data/raw_CT_98-401-X2016043_English_CSV_data.csv'
    #csv_path='data/extracted_CT2.csv'
    # extract(source,csv_path,info)

    CT_mapping_path='pednet_points/mapping.txt'
    #out='data/population.csv'
    #read_population(csv_path, CT_mapping_path, out)


    assign_avg('data/population.csv', CT_mapping_path, 'pednet_points/ends/avg_start_point.csv')




