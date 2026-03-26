# FEATURE ENGINEERING FUNCTIONS
from snowflake.snowpark import DataFrame
import snowflake.snowpark.functions as F

# Feature Engineering Functions
def uc01_load_data(customer_data: DataFrame, behavior_data: DataFrame) -> DataFrame:
    """
    Merges order, linetime and order_returns data and replaces Nulls/None with appropriate default values.
    customer_data      : A dataframe referencing the "CUSTOMER" table in the relevant schema
    behavior_data      : A dataframe referencing the "PURCHASE_BEHAVIOR" table in the relevant schema

    Returns            : Merged/cleansed dataframe with required columns
    """
    # Merge two dataframes
    raw_data =  customer_data.join(
            behavior_data,
            (customer_data["CUSTOMER_ID"] == behavior_data["CUSTOMER_ID"]),
            "left"
        )\
        .rename(
            {
                customer_data["UPDATED_AT"]: "CUSTOMER_UPDATED_AT",
                customer_data["CUSTOMER_ID"]: "CUSTOMER_ID",
                behavior_data["UPDATED_AT"]: "BEHAVIOR_UPDATED_AT"
            })

    return raw_data[[
        "CUSTOMER_ID",
        "AGE", 
        "GENDER",
        "STATE",
        "ANNUAL_INCOME", 
        "LOYALTY_TIER", 
        "TENURE_MONTHS", 
        "SIGNUP_DATE", 
        "CUSTOMER_UPDATED_AT", 
        "AVG_ORDER_VALUE", 
        "PURCHASE_FREQUENCY", 
        "RETURN_RATE", 
        "LIFETIME_VALUE", 
        "LAST_PURCHASE_DATE", 
        "TOTAL_ORDERS", 
        "BEHAVIOR_UPDATED_AT"
    ]]

def uc01_pre_process(data: DataFrame) -> DataFrame:
    """
    Performs model-agnostic Feature-Engineering to prepare data for model input for Customer Entity level features
    data         : A dataframe containing the merged/cleansed data from CUSTOMERS and PURCHASE_BEHAVIOR tables
    result       : Customer level model input features
    """
    # Round annual income to no decimals
    data = data\
        .with_column("ANNUAL_INCOME", F.round(F.col("ANNUAL_INCOME"), 0))
    
    # Generate new features from existing columns
    data = data.with_columns([
        "AVERAGE_ORDER_PER_MONTH",
        "DAYS_SINCE_LAST_PURCHASE", 
        "DAYS_SINCE_SIGNUP",
        "EXPECTED_DAYS_BETWEEN_PURCHASES",
        "DAYS_SINCE_EXPECTED_LAST_PURCHASE_DATE",
    ], [
        F.col("TOTAL_ORDERS") / F.col("TENURE_MONTHS"),
        F.datediff("day", F.col("LAST_PURCHASE_DATE"), F.col("BEHAVIOR_UPDATED_AT")),
        F.datediff("day", F.col("SIGNUP_DATE"), F.col("BEHAVIOR_UPDATED_AT")),
        F.lit(30) / F.col("PURCHASE_FREQUENCY"),
        F.round(F.datediff("day", F.col("LAST_PURCHASE_DATE"), F.col("BEHAVIOR_UPDATED_AT")) - F.lit((F.lit(30) / F.col("PURCHASE_FREQUENCY"))),0)
    ])

    return data
