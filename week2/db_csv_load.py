import pandas as pd
from sqlalchemy import create_engine

user = 'postgres'
pwd = '436025tn014' # Replace with your password
host = 'localhost'
port = '5432'
db = 'postgres'

engine = create_engine('postgresql://' + user + ':' + pwd + 
                       '@' + host + ':' + port + '/' + db)

vehicles = pd.read_csv('https://tf-assets-prod.s3.amazonaws.com/tf-curric/data-science/vehicles.csv')
vehicles.to_sql('vehicles', engine, index=False)

houseprices = pd.read_csv('https://tf-assets-prod.s3.amazonaws.com/tf-curric/data-science/houseprices.csv')
houseprices.to_sql('houseprices', engine, index=False)