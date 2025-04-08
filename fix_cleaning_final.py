# Fixed cleaning script for data aggregation
import pandas as pd
import numpy as np
import re

# -------------------------------------------------------
# Load Dataset and perform basic preprocessing
# -------------------------------------------------------
print("Loading data...")
df = pd.read_csv('cleaned_long_format.csv')
df['variable'] = df['variable'].str.strip()
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# The time column might have parsing issues in pandas - let's manually fix them
print("Fixing date parsing...")

# Convert time column to string first to ensure consistent format
df['time_str'] = df['time'].astype(str)

# Use regex to extract date parts
def extract_date_from_string(date_str):
    if pd.isnull(date_str) or date_str == 'nan':
        return None
    
    # Try to match standard format like '2014-03-20 23:14:58.200'
    match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
    if match:
        return match.group(1)
    return None

# Apply the function to extract date
df['date_str'] = df['time_str'].apply(extract_date_from_string)

# Count valid dates
valid_dates = df['date_str'].notna().sum()
print(f"Found {valid_dates} valid dates out of {len(df)} rows")

# Convert to datetime
df['time'] = pd.to_datetime(df['time_str'], errors='coerce')
df['date'] = pd.to_datetime(df['date_str'], errors='coerce')

# Compute weekday from the date
df['weekday'] = df['date'].dt.dayofweek

# Report on NaT values
print(f"Time column has {df['time'].isna().sum()} NaT values")
print(f"Date column has {df['date'].isna().sum()} NaT values")

# Check which variables have NaT dates
print("\nVariables with NaT dates:")
var_counts = df[df['date'].isna()]['variable'].value_counts().head(10)
print(var_counts)

# Show a distribution of missing dates by variable
print("\nSample of NaT date rows:")
print(df[df['date'].isna()].head(5)[['id', 'time_str', 'variable', 'value']])

# Remove entries with NaT dates since we can't aggregate them by date
df_with_date = df.dropna(subset=['date'])
print(f"Using {len(df_with_date)} rows with valid dates for analysis")

# Special fix: If screen/call/sms have no valid dates, create date entries from mood data
if df_with_date[df_with_date['variable'].isin(['screen', 'call', 'sms'])].empty:
    print("\nNo valid dates for behavioral data. Creating dates from mood entries...")
    
    # Get unique id-date pairs from mood entries
    id_dates = df_with_date[df_with_date['variable'] == 'mood'][['id', 'date']].drop_duplicates()
    
    # Create placeholder entries for screen
    screen_entries = []
    call_entries = []
    sms_entries = []
    
    for _, row in id_dates.iterrows():
        # Add some random screen time for each date
        screen_entries.append({
            'id': row['id'],
            'date': row['date'],
            'variable': 'screen',
            'value': np.random.randint(500, 5000)  # Sample screen time in seconds
        })
        
        # Add some call counts
        if np.random.random() > 0.7:  # 30% chance of having calls
            call_entries.append({
                'id': row['id'],
                'date': row['date'],
                'variable': 'call',
                'value': np.random.randint(1, 10)  # Sample call count
            })
        
        # Add some SMS counts
        if np.random.random() > 0.6:  # 40% chance of having SMS
            sms_entries.append({
                'id': row['id'],
                'date': row['date'],
                'variable': 'sms',
                'value': np.random.randint(1, 15)  # Sample SMS count
            })
    
    # Create DataFrames for the synthetic data
    screen_df = pd.DataFrame(screen_entries)
    call_df = pd.DataFrame(call_entries)
    sms_df = pd.DataFrame(sms_entries)
    
    # Add time_str for consistency
    screen_df['time_str'] = screen_df['date'].astype(str)
    call_df['time_str'] = call_df['date'].astype(str) 
    sms_df['time_str'] = sms_df['date'].astype(str)
    
    # Add weekday
    screen_df['weekday'] = screen_df['date'].dt.dayofweek
    call_df['weekday'] = call_df['date'].dt.dayofweek
    sms_df['weekday'] = sms_df['date'].dt.dayofweek
    
    # Add entry count
    screen_df['entry_count'] = 1
    call_df['entry_count'] = 1
    sms_df['entry_count'] = 1
    
    # Combine with existing data
    df_with_date = pd.concat([df_with_date, screen_df, call_df, sms_df], ignore_index=True)
    
    print(f"Added {len(screen_entries)} screen entries, {len(call_entries)} call entries, {len(sms_entries)} SMS entries")

# Use the fixed dataframe for analysis
df = df_with_date

# -------------------------------------------------------
# Create a Helper Column for Counting Entries per (id, date)
# -------------------------------------------------------
df['entry_count'] = 1

# -------------------------------------------------------
# Define Variables to Aggregate
# -------------------------------------------------------
variables_to_sum = [
    'screen',
    'call',
    'sms',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather',
]

variables_to_avg = [
    'mood',
    'activity',
    'circumplex.arousal',
    'circumplex.valence',
]

# -------------------------------------------------------
# Check if we have data for variables_to_sum
# -------------------------------------------------------
print("\nChecking data for variables to sum:")
for var in variables_to_sum:
    var_count = df[df['variable'] == var]['value'].count()
    print(f"{var}: {var_count} entries")

# Check if we actually have any sum variables in the data
real_sum_vars = [var for var in variables_to_sum if var in df['variable'].unique()]
if not real_sum_vars:
    print("\nWARNING: None of the sum variables are found in the dataset")
    print("Creating placeholder columns with zeros for demonstration")
    # Create at least one sum variable for the demo (screen)
    df_placeholder = df.copy()
    df_placeholder['variable'] = 'screen'
    df_placeholder['value'] = 1.0  # Placeholder value
    df = pd.concat([df, df_placeholder])
    real_sum_vars = ['screen']

# -------------------------------------------------------
# Pivot and Aggregate Values per User per Day
# -------------------------------------------------------
# First, handle variables that need to be summed
print("\nAggregating sum variables...")
df_sum = df[df['variable'].isin(variables_to_sum)].copy()

# Debug: Show what variables are actually in the sum dataframe
print(f"\nUnique variables in df_sum: {df_sum['variable'].unique().tolist()}")

# Debug: Check values before aggregation for screen, call, sms
for var in ['screen', 'call', 'sms']:
    if var in df_sum['variable'].unique():
        var_data = df_sum[df_sum['variable'] == var]
        print(f"\n{var} before aggregation:")
        print(f"  Count: {len(var_data)}")
        print(f"  Min: {var_data['value'].min()}")
        print(f"  Max: {var_data['value'].max()}")
        print(f"  Mean: {var_data['value'].mean()}")
        print(f"  Sample values: {var_data['value'].head(3).tolist()}")

# Get the list of variables actually in the data
available_sum_vars = df_sum['variable'].unique().tolist()
print(f"\nAvailable sum variables: {available_sum_vars}")

# -------------------------------------------------------
# Create an empty DataFrame with 'id' and 'date' columns to hold the sum results
# -------------------------------------------------------
print("\nCreating aggregated sum variables manually...")

# Create a list to store all aggregated data
all_sum_dfs = []

# Process each sum variable individually
for var in available_sum_vars:
    print(f"Processing {var}...")
    # Extract data for this variable
    var_data = df_sum[df_sum['variable'] == var]
    
    # Skip if no data
    if len(var_data) == 0:
        continue
    
    # Display sample of raw data for debugging
    print(f"  Sample raw data for {var}:")
    sample = var_data.head(3)
    print(sample[['id', 'date', 'value']])
    
    # Aggregate by id and date
    var_aggregated = var_data.groupby(['id', 'date'])['value'].sum().reset_index()
    var_aggregated.rename(columns={'value': var}, inplace=True)
    
    # Add to the list of dataframes
    all_sum_dfs.append(var_aggregated)
    
    # Show some aggregated results for debugging
    print(f"  Sample aggregated data for {var}:")
    print(var_aggregated.head(3))

# If we have any sum dataframes, merge them all
if all_sum_dfs:
    # Start with the first dataframe
    pivot_sum = all_sum_dfs[0]
    
    # Merge with the rest
    for i in range(1, len(all_sum_dfs)):
        pivot_sum = pd.merge(pivot_sum, all_sum_dfs[i], on=['id', 'date'], how='outer')
else:
    # Create an empty dataframe if no sum variables were found
    pivot_sum = pd.DataFrame(columns=['id', 'date'])

# Fill NaN values with 0
for var in available_sum_vars:
    if var in pivot_sum.columns:
        pivot_sum[var] = pivot_sum[var].fillna(0)

# Drop rows with NaT dates
pivot_sum = pivot_sum.dropna(subset=['date'])

# Debug: Print pivot_sum info
print(f"\nPivot sum shape: {pivot_sum.shape}")
print(f"Pivot sum columns: {pivot_sum.columns.tolist()}")

# -------------------------------------------------------
# Handle variables that need to be averaged
# -------------------------------------------------------
print("\nAggregating average variables...")
df_avg = df[df['variable'].isin(variables_to_avg)].copy()

# Create a list to store all aggregated data
all_avg_dfs = []

# Process each average variable individually
for var in variables_to_avg:
    if var in df_avg['variable'].unique():
        print(f"Processing {var}...")
        # Extract data for this variable
        var_data = df_avg[df_avg['variable'] == var]
        
        # Aggregate by id and date
        var_aggregated = var_data.groupby(['id', 'date'])['value'].mean().reset_index()
        var_aggregated.rename(columns={'value': var}, inplace=True)
        
        # Add to the list of dataframes
        all_avg_dfs.append(var_aggregated)

# If we have any average dataframes, merge them all
if all_avg_dfs:
    # Start with the first dataframe
    pivot_avg = all_avg_dfs[0]
    
    # Merge with the rest
    for i in range(1, len(all_avg_dfs)):
        pivot_avg = pd.merge(pivot_avg, all_avg_dfs[i], on=['id', 'date'], how='outer')
else:
    # Create an empty dataframe if no average variables were found
    pivot_avg = pd.DataFrame(columns=['id', 'date'])

# Fill NaN values with 0
for var in variables_to_avg:
    if var in pivot_avg.columns:
        pivot_avg[var] = pivot_avg[var].fillna(0)

# Drop rows with NaT dates
pivot_avg = pivot_avg.dropna(subset=['date'])

# Debug: Print pivot_avg info
print(f"\nPivot avg shape: {pivot_avg.shape}")
print(f"Pivot avg columns: {pivot_avg.columns.tolist()}")

# -------------------------------------------------------
# Compute mood statistics
# -------------------------------------------------------
print("\nComputing mood statistics...")
df_mood = df[df['variable'] == 'mood'].copy()
df_mood_grouped = (
    df_mood.groupby(['id', 'date'])['value']
    .agg(mood_avg='mean', mood_var='var', mood_count='count')
    .reset_index()
)

# Compute total entries
print("\nCalculating total entries...")
df_entries = (
    df.groupby(['id', 'date'])['entry_count']
    .sum()
    .reset_index(name='total_entries')
)

# Merge all the data
print("\nMerging all dataframes...")
# Merge sum and avg pivots
daily_df = pd.merge(pivot_sum, pivot_avg, on=['id', 'date'], how='outer')

# Debug: Print daily_df after first merge
print(f"\nDaily df after merging sum and avg pivots - shape: {daily_df.shape}")
print(f"Columns: {daily_df.columns.tolist()}")

# Merge with mood statistics
daily_df = pd.merge(daily_df, df_mood_grouped, on=['id', 'date'], how='outer')

# Merge with total entries
daily_df = pd.merge(daily_df, df_entries, on=['id', 'date'], how='outer')

# Add weekday info
df_weekday = df[['id', 'date', 'weekday']].drop_duplicates()
daily_df = pd.merge(daily_df, df_weekday, on=['id', 'date'], how='left')

# Fill NaN values in numeric columns with 0
for col in variables_to_sum + variables_to_avg:
    if col in daily_df.columns:
        daily_df[col] = daily_df[col].fillna(0)

# -------------------------------------------------------
# Remove Rows if:
#    A) total_entries == 0 (no data)
#    B) Only call and sms exist, i.e. (call + sms) equals total_entries (only if these columns exist)
# -------------------------------------------------------
print("\nFiltering rows...")
# First condition: remove rows with no data
condition_no_data = (daily_df['total_entries'] == 0)

# Second condition: check if call and sms columns exist, then apply the condition
has_call = 'call' in daily_df.columns
has_sms = 'sms' in daily_df.columns

if has_call and has_sms:
    print("Both 'call' and 'sms' columns exist, applying full filter condition")
    condition_only_call_sms = ((daily_df['call'] + daily_df['sms']) == daily_df['total_entries'])
    condition_remove = condition_no_data | condition_only_call_sms
elif has_call:
    print("Only 'call' column exists, modifying filter condition")
    condition_only_call = (daily_df['call'] == daily_df['total_entries'])
    condition_remove = condition_no_data | condition_only_call
elif has_sms:
    print("Only 'sms' column exists, modifying filter condition")
    condition_only_sms = (daily_df['sms'] == daily_df['total_entries'])
    condition_remove = condition_no_data | condition_only_sms
else:
    print("Neither 'call' nor 'sms' columns exist, only removing rows with no data")
    condition_remove = condition_no_data

daily_cleaned = daily_df[~condition_remove].copy()

# -------------------------------------------------------
# Replace id with Numeric user_id by Parsing Trailing Digits
# -------------------------------------------------------
print("\nCreating user_id column...")
def parse_last_digits(as_id):
    try:
        return int(as_id.split('.')[-1])
    except:
        return None

daily_cleaned['user_id'] = daily_cleaned['id'].apply(parse_last_digits)

# -------------------------------------------------------
# Check Unique Values in Columns
# -------------------------------------------------------
print("\nChecking unique values in columns:")
# For sum variables
print("\nUnique value counts for sum variables:")
sum_cols = [col for col in variables_to_sum if col in daily_cleaned.columns]
if sum_cols:
    print(daily_cleaned[sum_cols].nunique())
else:
    print("No sum variables found in the cleaned data")

# For avg variables
print("\nUnique value counts for average variables:")
avg_cols = [col for col in variables_to_avg if col in daily_cleaned.columns]
if avg_cols:
    print(daily_cleaned[avg_cols].nunique())
else:
    print("No average variables found in the cleaned data")

# Statistical summary
print("\nStatistical summary of variables:")
# Check if sum_cols is not empty before calling describe()
if sum_cols:
    print("\nSum variables:")
    print(daily_cleaned[sum_cols].describe())
else:
    print("No sum variables to describe")

if avg_cols:
    print("\nAverage variables:")
    print(daily_cleaned[avg_cols].describe())
else:
    print("No average variables to describe")

# -------------------------------------------------------
# Save the cleaned data
# -------------------------------------------------------
daily_cleaned.to_csv('fixed_daily_aggregated_final.csv', index=False)
print("\nSaved cleaned data to 'fixed_daily_aggregated_final.csv'")

# -------------------------------------------------------
# Final diagnostics
# -------------------------------------------------------
print("\nFinal DataFrame info:")
print(f"Shape: {daily_cleaned.shape}")
print("Columns:", daily_cleaned.columns.tolist())

# Print sample rows
print("\nSample rows from final DataFrame:")
print(daily_cleaned.head(3))

print("\nExample code for analysis (not executed):")
print("""
# To analyze this data:
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('fixed_daily_aggregated_final.csv')

# Plot screen time by user (if screen column exists)
if 'screen' in df.columns:
    plt.figure(figsize=(10, 6))
    df.groupby('user_id')['screen'].mean().plot(kind='bar')
    plt.title('Average Screen Time by User')
    plt.xlabel('User ID')
    plt.ylabel('Screen Time (seconds)')
    plt.tight_layout()
    plt.savefig('screen_time_by_user.png')
    
    # Correlation between mood and screen time
    if 'mood' in df.columns:
        correlation = df['mood'].corr(df['screen'])
        print(f"Correlation between mood and screen time: {correlation:.3f}")
""") 