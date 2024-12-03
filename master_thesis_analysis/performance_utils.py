import pandas as pd

def prepare_performance_data(df):
    """
    Prepares performance data by calculating outbound performance and cleaning the dataset.
    """
    # Get only finished trials
    finished_df = df[df['complete_fit'].notna()].copy()
    
    # Initialize outbound performance column
    finished_df['outbound_performance'] = None
    
    # Calculate outbound performance for each row
    for idx, row in finished_df.iterrows():
        if 'Outbound' in row['task']:
            _calculate_outbound_performance(finished_df, idx, row)
    
    # Clean the data
    columns_to_check = ['complete_fit', 'from_well', 'to_well', 'task', 'is_correct', 'turn']
    cleaned_df = remove_empty_list_entries(finished_df, columns_to_check)
    
    return cleaned_df

def _calculate_outbound_performance(df, idx, row):
    """Helper function to calculate outbound performance for a single row."""
    tasks = row['task']
    is_correct = [True if str(x).lower() == 'true' else False for x in row['is_correct']]
    
    correct_outbound_count = sum([tasks[i] == 'Outbound' and is_correct[i] == True 
                                for i in range(len(tasks))])
    total_outbound_count = tasks.count('Outbound')
    
    if total_outbound_count > 0:
        df.at[idx, 'outbound_performance'] = correct_outbound_count / total_outbound_count

def remove_empty_list_entries(df, columns):
    """Remove rows with empty lists in specified columns."""
    def is_valid_row(row):
        return all(len(row[col]) > 0 for col in columns if isinstance(row[col], list))
    return df[df.apply(is_valid_row, axis=1)]

def remove_partial_entries(row):
    """Remove partial entries from a row."""
    if all(isinstance(row[col], list) for col in ['complete_fit', 'from_well', 'to_well', 'task', 'is_correct', 'turn']):
        indices_to_keep = [i for i, fit in enumerate(row['complete_fit']) if fit != 'partial']
    
        try:
            if indices_to_keep:
                for col in ['complete_fit', 'from_well', 'to_well', 'task', 'is_correct', 'turn']:
                    row[col] = [row[col][i] for i in indices_to_keep]
            else:
                for col in ['complete_fit', 'from_well', 'to_well', 'task', 'is_correct', 'turn']:
                    row[col] = []
        except:
            pass
    
    return row 