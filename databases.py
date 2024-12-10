from sqlalchemy import create_engine
import pandas as pd

# Import username and password from keys
from keys import post_username, post_password

# Function to process data
def process_data():
    # Database connection parameters
    dbname = "team2_project4"
    host = "localhost"
    port = "5432"

    # Create SQLAlchemy engine
    engine = create_engine(f'postgresql://{post_username}:{post_password}@{host}:{port}/{dbname}')

    # Query to fetch data
    query = "SELECT * FROM heart_failure_records;"

    # Read data into a DataFrame
    df = pd.read_sql(query, engine)
    return df

# Call the function
df = process_data()

