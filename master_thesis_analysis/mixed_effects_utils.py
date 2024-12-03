import pandas as pd
import statsmodels.formula.api as smf
import warnings
import numpy as np

# Suppress all warnings
warnings.filterwarnings('ignore')

def fit_mixed_models_between_task(df):
    """
    Fits mixed linear models for branching factor and tau with styled output.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data to analyze
        
    Returns:
    --------
    dict
        Contains model results and styled table
    """


    df['state'] = df['state'].replace({'wake': 'working memory task', 'sleep': 'control'})

    # Fit mixed linear model for branching_factor
    model_bf = smf.mixedlm("branching_factor ~ state * area", df, groups=df["animal"])
    result_bf = model_bf.fit()
    
    # Fit mixed linear model for tau
    model_tau = smf.mixedlm("tau ~ state * area", df, groups=df["animal"]) 
    result_tau = model_tau.fit()
    
    # Create formatted results for both models
    def format_model_results(result, measure_name):
        formatted_data = []
        
        # Extract fixed effects
        for param in result.params.index:
            formatted_data.append({
                'Measure': measure_name,
                'Parameter': param,
                'Coefficient': f"{result.params[param]:.3f}",
                'Std. Error': f"{result.bse[param]:.3f}",
                'z-value': f"{result.tvalues[param]:.3f}",
                'p-value': f"{result.pvalues[param]:.3e}",
                'CI 95%': f"[{result.conf_int().iloc[result.params.index.get_loc(param), 0]:.3f}, "
                         f"{result.conf_int().iloc[result.params.index.get_loc(param), 1]:.3f}]"
            })
        
        # Add random effects information
        re_var = float(result.cov_re.iloc[0,0])  # Extract random effect variance
        formatted_data.append({
            'Measure': measure_name,
            'Parameter': 'Random Effect (Animal)',
            'Coefficient': f"{np.sqrt(re_var):.3f}",
            'Std. Error': '--',
            'z-value': '--',
            'p-value': '--',
            'CI 95%': '--'
        })
        
        return formatted_data
    
    # Combine results from both models
    formatted_data = format_model_results(result_bf, 'Branching Factor')
    formatted_data.extend(format_model_results(result_tau, 'Neural Timescale'))
    
    # Create DataFrame
    table_df = pd.DataFrame(formatted_data)
    
    # Find index where measure changes
    measure_change_idx = table_df[table_df['Measure'] == 'Neural Timescale'].index[0]
    
    # Apply styling
    styled_df = (table_df.style
                .set_properties(**{
                    'text-align': 'center',
                    'font-size': '11pt',
                    'font-family': 'Arial',
                    'border': '2px solid black'
                })
                .set_table_styles([
                    {'selector': 'table',
                     'props': [('border', '2px solid black')]},
                    {'selector': 'th',
                     'props': [('background-color', '#f0f0f0'),
                             ('font-weight', 'bold'),
                             ('text-align', 'center'),
                             ('border-bottom', '2px solid black'),
                             ('white-space', 'pre-wrap')]},
                    {'selector': 'td',
                     'props': [('white-space', 'pre-wrap')]},
                    {'selector': 'caption',
                     'props': [('caption-side', 'top'),
                             ('font-size', '14pt'),
                             ('font-weight', 'bold'),
                             ('margin-bottom', '10px')]}
                ])
                .apply(lambda x: ['border-bottom: 2px solid black' 
                                if x.name == measure_change_idx - 1 
                                else '' for _ in x], axis=1)
                .set_caption('Mixed Effects Model Results (Between-Task)')
                .hide(axis='index'))
    
    # Display the styled table
    display(styled_df)
    
    return {
        'model_bf': result_bf,
        'model_tau': result_tau,
        'styled_results': styled_df
    }

def analyze_animals_mixed_effects_between_task(df):
    """
    Runs mixed effects models for each animal analyzing state and area effects.
    """
    warnings.filterwarnings('ignore')
    
    results_list = []
    n_animals = len(df['animal'].unique())
    
    for animal in df['animal'].unique():
        animal_df = df[df['animal'] == animal].copy()
        
        try:
            for dv in ['tau', 'branching_factor']:
                result = _fit_mixed_effects_model(animal_df, dv)
                results_list.extend(_process_model_results(result, animal, dv, n_animals))
        except Exception as e:
            print(f"Error processing animal {animal}: {str(e)}")
            continue
    
    return generate_mixed_effects_tables(pd.concat(results_list, ignore_index=True))

def _fit_mixed_effects_model(df, dependent_var):
    """Helper function to fit mixed effects model."""
    model = smf.mixedlm(
        f"{dependent_var} ~ state * area",
        data=df,
        groups=df["animal"]
    )
    return model.fit(reml=True)

def _process_model_results(result, animal, dv, n_animals):
    """Helper function to process model results."""
    coef_df = pd.DataFrame({
        'animal': animal,
        'dependent_var': dv,
        'predictor': result.params.index,
        'coefficient': result.params.values,
        'std_err': result.bse.values,
        'z_value': result.tvalues.values,
        'p_value': result.pvalues.values,
        'p_value_bonf': result.pvalues.values * n_animals,
        'ci_lower': result.conf_int()[0].values,
        'ci_upper': result.conf_int()[1].values
    })
    
    re_var = pd.DataFrame({
        'animal': animal,
        'dependent_var': dv,
        'predictor': ['Group Var'],
        'coefficient': [result.cov_re.iloc[0,0]],
        'std_err': [None],
        'z_value': [None],
        'p_value': [None],
        'p_value_bonf': [None],
        'ci_lower': [None],
        'ci_upper': [None]
    })
    
    return [coef_df, re_var] 

def fit_mixed_effects_model_within_task(df):
    """
    Fits mixed effects models and displays results in a styled table format.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: 'tau', 'outbound_performance', 'area', 'animal'
    
    Returns:
    --------
    dict with model results and styled tables
    """
    # Fit branching factor model
    model_m = smf.mixedlm(
        "branching_factor ~ outbound_performance * area", 
        data=df, 
        groups=df["animal"]
    )
    result_m = model_m.fit()
    
    # Fit tau model
    model_tau = smf.mixedlm(
        "tau ~ outbound_performance * area",
        data=df,
        groups=df["animal"]
    )
    result_tau = model_tau.fit()
    
    # Create formatted results for both models
    def format_model_results(result, measure_name):
        formatted_data = []
        
        # Extract fixed effects
        for param in result.params.index:
            formatted_data.append({
                'Measure': measure_name,
                'Parameter': param,
                'Coefficient': f"{result.params[param]:.3f}",
                'Std. Error': f"{result.bse[param]:.3f}",
                'z-value': f"{result.tvalues[param]:.3f}",
                'p-value': f"{result.pvalues[param]:.3e}",
                'CI 95%': f"[{result.conf_int().iloc[result.params.index.get_loc(param), 0]:.3f}, "
                         f"{result.conf_int().iloc[result.params.index.get_loc(param), 1]:.3f}]"
            })
        
        # Add random effects information
        re_var = float(result.cov_re.iloc[0,0])  # Extract random effect variance
        formatted_data.append({
            'Measure': measure_name,
            'Parameter': 'Random Effect (Animal)',
            'Coefficient': f"{np.sqrt(re_var):.3f}",
            'Std. Error': '--',
            'z-value': '--',
            'p-value': '--',
            'CI 95%': '--'
        })
        
        return formatted_data
    
    # Combine results from both models
    formatted_data = format_model_results(result_m, 'Branching Factor')
    formatted_data.extend(format_model_results(result_tau, 'Neural Timescale'))
    
    # Create DataFrame
    table_df = pd.DataFrame(formatted_data)
    
    # Find index where measure changes
    measure_change_idx = table_df[table_df['Measure'] == 'Neural Timescale'].index[0]
    
    # Apply styling
    styled_df = (table_df.style
                .set_properties(**{
                    'text-align': 'center',
                    'font-size': '11pt',
                    'font-family': 'Arial',
                    'border': '2px solid black'
                })
                .set_table_styles([
                    {'selector': 'table',
                     'props': [('border', '2px solid black')]},
                    {'selector': 'th',
                     'props': [('background-color', '#f0f0f0'),
                             ('font-weight', 'bold'),
                             ('text-align', 'center'),
                             ('border-bottom', '2px solid black'),
                             ('white-space', 'pre-wrap')]},
                    {'selector': 'td',
                     'props': [('white-space', 'pre-wrap')]},
                    {'selector': 'caption',
                     'props': [('caption-side', 'top'),
                             ('font-size', '14pt'),
                             ('font-weight', 'bold'),
                             ('margin-bottom', '10px')]}
                ])
                .apply(lambda x: ['border-bottom: 2px solid black' 
                                if x.name == measure_change_idx - 1 
                                else '' for _ in x], axis=1)
                .set_caption('Mixed Effects Model Results (Within-Task)')
                .hide(axis='index'))
    
    # Display the styled table
    display(styled_df)
    
    return {
        'model_m': result_m,
        'model_tau': result_tau,
        'styled_results': styled_df
    }

def analyze_animals_mixed_effects_within_task(df):
    """
    Runs mixed effects models for each animal and compiles results into a DataFrame.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Full DataFrame with all animals
        
    Returns:
    --------
    pd.DataFrame
        Compiled results for all animals
    """
    import statsmodels.formula.api as smf
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')  # Suppress convergence warnings
    
    # Initialize lists to store results
    results_list = []
    
    # Get unique animals
    animals = df['animal'].unique()
    
    for animal in animals:
        # Filter data for this animal
        animal_df = df[df['animal'] == animal].copy()
    
        
        try:
            # Fit models for both tau and branching factor
            for dv in ['tau', 'branching_factor']:
                # Fit the model
                model = smf.mixedlm(
                    f"{dv} ~ outbound_performance * area",
                    data=animal_df,
                    groups=animal_df["animal"]
                )
                
                result = model.fit(reml=True)
                
                # Extract coefficients and statistics
                coef_df = pd.DataFrame({
                    'animal': animal,
                    'dependent_var': dv,
                    'predictor': result.params.index,
                    'coefficient': result.params.values,
                    'std_err': result.bse.values,
                    'z_value': result.tvalues.values,
                    'p_value': result.pvalues.values,
                    'ci_lower': result.conf_int()[0].values,
                    'ci_upper': result.conf_int()[1].values
                })
                
                # Add random effects variance
                re_var = pd.DataFrame({
                    'animal': animal,
                    'dependent_var': dv,
                    'predictor': ['Group Var'],
                    'coefficient': [result.cov_re.iloc[0,0]],
                    'std_err': [None],
                    'z_value': [None],
                    'p_value': [None],
                    'ci_lower': [None],
                    'ci_upper': [None]
                })
                
                # Combine fixed and random effects
                results_list.append(coef_df)
                results_list.append(re_var)
                
        except Exception as e:
            print(f"Error processing animal {animal}: {str(e)}")
            continue
    
    # Combine all results
    results_df = pd.concat(results_list, ignore_index=True)
    
    # Format p-values
    return generate_mixed_effects_tables(results_df)

def generate_mixed_effects_tables(results_df):
    """
    Generates publication-ready styled DataFrames for each animal's mixed effects results.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from analyze_animals_mixed_effects_within_task
        
    Returns:
    --------
    dict
        Dictionary of styled DataFrames, one per animal
    """
    def format_value(value, is_p_value=False):
        """Helper function to format values"""
        if pd.isna(value):
            return '--'
        if is_p_value:
            if float(value) < 0.001:
                return '< 0.001'
            return f"{float(value):.3f}"
        return f"{float(value):.3f}"
    
    styled_tables = {}
    
    # Process each animal separately
    for animal in sorted(results_df['animal'].unique()):
        animal_data = results_df[results_df['animal'] == animal].copy()
        
        # Create formatted DataFrame
        formatted_data = []
        
        # Process each measure (tau and branching factor)
        for measure in ['tau', 'branching_factor']:
            measure_data = animal_data[animal_data['dependent_var'] == measure]
            measure_name = 'Neural Timescale' if measure == 'tau' else 'Branching Factor'
            
            for _, row in measure_data.iterrows():
                formatted_data.append({
                    'Measure': measure_name,
                    'Predictor': row['predictor'],
                    'Coefficient (SE)': f"{format_value(row['coefficient'])} ({format_value(row['std_err'])})",
                    'z-value': format_value(row['z_value']),
                    'p-value': format_value(row['p_value'], is_p_value=True),
                    'CI 95%': f"[{format_value(row['ci_lower'])}, {format_value(row['ci_upper'])}]"
                })
        
        # Create DataFrame
        table_df = pd.DataFrame(formatted_data)
        
        # Find the index where measure changes from Neural Timescale to Branching Factor
        measure_change_idx = table_df[table_df['Measure'] == 'Branching Factor'].index[0]
        
        # Apply styling
        styled_df = (table_df.style
                    .set_properties(**{
                        'text-align': 'center',
                        'font-size': '11pt',
                        'font-family': 'Arial',
                        'border': '2px solid black'  # Add black edge around whole table
                    })
                    .set_table_styles([
                        # Table borders
                        {'selector': 'table',
                         'props': [('border', '2px solid black')]},
                        # Header styling
                        {'selector': 'th',
                         'props': [('background-color', '#f0f0f0'),
                                 ('font-weight', 'bold'),
                                 ('text-align', 'center'),
                                 ('border-bottom', '2px solid black')]},
                        # Caption styling
                        {'selector': 'caption',
                         'props': [('caption-side', 'top'),
                                 ('font-size', '14pt'),  # Bigger caption
                                 ('font-weight', 'bold'),
                                 ('margin-bottom', '10px')]}
                    ])
                    # Add border between Neural Timescale and Branching Factor
                    .apply(lambda x: ['border-bottom: 4px solid black' 
                                    if x.name == measure_change_idx - 1 
                                    else '' for _ in x], axis=1)
                    .set_caption(f'Mixed Effects Model Results for Animal {animal}')
                    .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16pt')]}], overwrite=False)
                    .hide(axis='index'))
        
        # Add to dictionary
        styled_tables[animal] = styled_df
        
        # Print the table
        display(styled_df)
        print("\n")
    
    return styled_tables
