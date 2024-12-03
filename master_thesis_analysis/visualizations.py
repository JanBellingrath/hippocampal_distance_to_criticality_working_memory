import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def plot_animal_specific_between_task_analysis(df, figsize=(20, 10)):
    """
    Creates combined state distribution and interaction plots for each animal.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing all required columns
    figsize : tuple
        Size of the combined figure for each animal
    """
    # Get unique animals
    animals = df['animal'].unique()
    

    df['state'] = df['state'].replace({'wake': 'working memory task', 'sleep': 'control'})

    # Define order for states
    state_order = ['control', 'working memory task']
    
    for animal in animals:
        # Filter data for this animal
        animal_df = df[df['animal'] == animal].copy()
        
        # Drop NaN values in 'tau' and 'state' columns
        animal_df = animal_df.dropna(subset=['tau', 'state'])
        
        # Ensure states are ordered and categorical
        animal_df['state'] = pd.Categorical(
            animal_df['state'],
            categories=state_order,
            ordered=True
        )
        
        # Skip this animal if no valid data
        if len(animal_df) == 0:
            print(f"Skipping animal {animal} - no valid data")
            continue
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=figsize)
        
        # Add GridSpec to create 2 rows of plots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # First row: State distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot state distributions in first row
        sns.violinplot(data=animal_df, x='state', y='tau', ax=ax1,
                      inner=None, palette='viridis', order=state_order)
        sns.pointplot(data=animal_df, x='state', y='tau', ax=ax1,
                     estimator=np.median, color='#21918c',
                     markers='s', markersize=5, linewidth=1,
                     order=state_order)
        
        ax1.set_title('Neural Timescale by State', fontsize=14)
        ax1.set_xlabel('State', fontsize=12)
        ax1.set_ylabel('Intrinsic Neural Timescale (ms)', fontsize=12)
        ax1.set_ylim(bottom=0)
        ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
        
        sns.violinplot(data=animal_df, x='state', y='branching_factor', ax=ax2,
                      inner=None, palette='viridis', order=state_order)
        sns.pointplot(data=animal_df, x='state', y='branching_factor', ax=ax2,
                     estimator=np.median, color='#21918c',
                     markers='s', markersize=5, linewidth=1,
                     order=state_order)
        
        ax2.set_title('Branching Factor by State', fontsize=14)
        ax2.set_xlabel('State', fontsize=12)
        ax2.set_ylabel('Branching Factor', fontsize=12)
        ax2.set_ylim(bottom=0.85, top=1)
        ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
        
        # Second row: Area-State interactions
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        sns.pointplot(data=animal_df, x='state', y='tau', hue='area', 
                     ax=ax3, palette='viridis',
                     markers=['o', 's'], linestyles=['-', '--'],
                     capsize=.1, errwidth=1,
                     estimator=np.mean,
                     order=state_order)
        
        ax3.set_title('Neural Timescale by State and Area', fontsize=14)
        ax3.set_xlabel('State', fontsize=12)
        ax3.set_ylabel('Intrinsic Neural Timescale (ms)', fontsize=12)
        ax3.legend(title='Area', bbox_to_anchor=(1.05, 1))
        ax3.set_xticklabels(ax3.get_xticklabels(), ha='right')
        
        sns.pointplot(data=animal_df, x='state', y='branching_factor', hue='area',
                     ax=ax4, palette='viridis',
                     markers=['o', 's'], linestyles=['-', '--'],
                     capsize=.1, errwidth=1,
                     estimator=np.mean,
                     order=state_order)
        
        ax4.set_title('Branching Factor by State and Area', fontsize=14)
        ax4.set_xlabel('State', fontsize=12)
        ax4.set_ylabel('Branching Factor', fontsize=12)
        ax4.legend(title='Area', bbox_to_anchor=(1.05, 1))
        ax4.set_xticklabels(ax4.get_xticklabels(), ha='right')
        
        plt.suptitle(f'Animal: {animal}', size=30, y=1.05, weight='bold')
        plt.tight_layout()
        plt.show()

def plot_state_distributions(df, figsize=(12, 5)):
    """
    Creates professional violin plots for tau and branching factor across states.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'tau', 'branching_factor', and 'state' columns
    figsize : tuple
        Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    
    df['state'] = df['state'].replace({'wake': 'working memory task', 'sleep': 'control'})

    # Plot tau
    sns.violinplot(data=df, x='state', y='tau', ax=ax1,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='state', y='tau', ax=ax1,
                 estimator=np.median, color='#21918c',
                 markers='s', markersize=5, linewidth=1)
    
    # Customize tau plot
    ax1.set_title('Neural Timescale by State')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Intrinsic Neural Timescale (ms)')
    ax1.set_ylim(bottom=0, top=3000)  # Truncate at 0
    # Plot branching factor
    sns.violinplot(data=df, x='state', y='branching_factor', ax=ax2,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='state', y='branching_factor', ax=ax2,
                 estimator=np.median, color='#21918c',
                 markers='s', markersize=5, linewidth=1)
    # Customize branching factor plot
    ax2.set_title('Branching Factor by State')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Branching Factor')
    ax2.set_ylim(bottom=0.85, top=1)  # Truncate at 0 and 1
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='black', label='Median',
               markerfacecolor='black', markersize=5, linestyle='-', linewidth=1)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_animal_distributions(df, figsize=(15, 6)):
    """
    Creates professional violin plots for tau and branching factor across animals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'tau', 'branching_factor', and 'animal' columns
    figsize : tuple
        Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot tau
    sns.violinplot(data=df, x='animal', y='tau', ax=ax1,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='animal', y='tau', ax=ax1,
                 estimator=np.median, color='black',
                 markers='s', markersize=5, linewidth=1)
    
    # Customize tau plot
    ax1.set_title('Neural Timescale by Animal')
    ax1.set_xlabel('Animal ID')
    ax1.set_ylabel('Intrinsic Neural Timescale (ms)')
    ax1.set_ylim(0, 3000)  # Set y-axis limits
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    
    # Plot branching factor
    sns.violinplot(data=df, x='animal', y='branching_factor', ax=ax2,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='animal', y='branching_factor', ax=ax2,
                 estimator=np.median, color='black',
                 markers='s', markersize=5, linewidth=1)
    
    # Customize branching factor plot
    ax2.set_title('Branching Factor by Animal')
    ax2.set_xlabel('Animal ID')
    ax2.set_ylabel('Branching Factor')
    ax2.set_ylim(0.85, 1)  # Set y-axis limits
    ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='black', label='Median',
               markerfacecolor='black', markersize=5, linestyle='-', linewidth=1)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    return fig

def plot_area_state_interactions(df, figsize=(12, 5)):
    """
    Creates professional interaction plots for tau and branching factor across states and areas.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'tau', 'branching_factor', 'state', and 'area' columns
    figsize : tuple
        Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create mapping dictionary for state labels
    state_mapping = {'wake': 'task', 'sleep': 'control'}
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].map(state_mapping)
    
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot tau interaction
    sns.pointplot(data=df, x='state', y='tau', hue='area', 
                 ax=ax1, palette='viridis',
                 markers=['o', 's'], linestyles=['-', '--'],
                 capsize=.1, errwidth=1,
                 estimator=np.mean)
    
    # Customize tau plot
    ax1.set_title('Neural Timescale by State and Area')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Intrinsic Neural Timescale (ms)')
    ax1.legend(title='Area', bbox_to_anchor=(1.05, 1))
    
    # Plot branching factor interaction
    sns.pointplot(data=df, x='state', y='branching_factor', hue='area',
                 ax=ax2, palette='viridis',
                 markers=['o', 's'], linestyles=['-', '--'],
                 capsize=.1, errwidth=1,
                 estimator=np.mean)
    
    # Customize branching factor plot
    ax2.set_title('Branching Factor by State and Area')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Branching Factor')
    ax2.legend(title='Area', bbox_to_anchor=(1.05, 1))
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_outbound_performance(df, figsize=(10, 6)):
    """
    Creates a visualization of outbound task performance distribution.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'outbound_performance' column
    figsize : tuple
        Figure size (width, height)
    """
    # Input validation
    if 'outbound_performance' not in df.columns:
        raise ValueError("DataFrame must contain 'outbound_performance' column")
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create main plot
    g = sns.countplot(data=df,
                      x='outbound_performance',
                      order=[0.0, 0.5, 1.0],
                      palette='viridis')
    
    # Add percentage labels on top of bars
    total = len(df['outbound_performance'].dropna())
    for p in g.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        g.annotate(percentage,
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center',
                  va='bottom')
    
    # Customize plot
    plt.xlabel('Performance Score')
    plt.ylabel('Count')
    plt.title('Distribution of Outbound Task Performance')
    
    # Add summary statistics as text
    summary_stats = (f"n = {total}\n"
                    f"Mean = {df['outbound_performance'].mean():.3f}\n"
                    f"Median = {df['outbound_performance'].median():.3f}")
    
    
    plt.tight_layout()
    return plt.gcf()

def plot_performance_distributions(df, figsize=(12, 5)):
    """
    Creates professional violin plots for tau and branching factor across performance categories.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'tau', 'branching_factor', and 'outbound_performance' columns
    figsize : tuple
        Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot tau
    sns.violinplot(data=df, x='outbound_performance', y='tau', ax=ax1,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='outbound_performance', y='tau', ax=ax1,
                 estimator=np.mean, color='#21918c',
                 markers='s', markersize=5, linewidth=1)
    
    # Customize tau plot
    ax1.set_title('Neural Timescale by Performance')
    ax1.set_xlabel('Performance Score')
    ax1.set_ylabel('Intrinsic Neural Timescale (ms)')
    ax1.set_ylim(bottom=0)  # Truncate at 0
    
    # Plot branching factor
    sns.violinplot(data=df, x='outbound_performance', y='branching_factor', ax=ax2,
                  inner=None, palette='viridis')
    sns.pointplot(data=df, x='outbound_performance', y='branching_factor', ax=ax2,
                 estimator=np.mean, color='#21918c',
                 markers='s', markersize=5, linewidth=1)
    
    # Customize branching factor plot
    ax2.set_title('Branching Factor by Performance')
    ax2.set_xlabel('Performance Score')
    ax2.set_ylabel('Branching Factor')
    ax2.set_ylim(bottom=0.85, top=1)  # Set reasonable limits
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='black', label='Median',
               markerfacecolor='black', markersize=5, linestyle='-', linewidth=1)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_area_performance_interaction(df, figsize=(12, 5)):
    """
    Creates professional interaction plots for tau and branching factor across performance levels and areas.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'tau', 'branching_factor', 'outbound_performance', and 'area' columns
    figsize : tuple
        Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define order for performance categories
    perf_order = ['0.0', '0.5', '1.0']
    
    # Plot tau interaction
    sns.pointplot(data=df, x='outbound_performance', y='tau', hue='area', 
                 ax=ax1, palette='viridis',
                 markers=['o', 's'], linestyles=['-', '--'],
                 capsize=.1, errwidth=1,
                 estimator=np.mean,
                 order=perf_order)
    
    # Customize tau plot
    ax1.set_title('Neural Timescale by Performance and Area')
    ax1.set_xlabel('Performance Score')
    ax1.set_ylabel('Intrinsic Neural Timescale (ms)')
    ax1.legend(title='Area', bbox_to_anchor=(1.05, 1))
    
    # Plot branching factor interaction
    sns.pointplot(data=df, x='outbound_performance', y='branching_factor', hue='area',
                 ax=ax2, palette='viridis',
                 markers=['o', 's'], linestyles=['-', '--'],
                 capsize=.1, errwidth=1,
                 estimator=np.mean,
                 order=perf_order)
    
    # Customize branching factor plot
    ax2.set_title('Branching Factor by Performance and Area')
    ax2.set_xlabel('Performance Score')
    ax2.set_ylabel('Branching Factor')
    ax2.legend(title='Area', bbox_to_anchor=(1.05, 1))
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_animal_specific_analysis(df, figsize=(20, 10)):
    """
    Creates combined performance distribution and interaction plots for each animal.
    """
    # Get unique animals
    animals = df['animal'].unique()
    
    # Define order for performance categories and ensure they're strings
    perf_order = ['0.0', '0.5', '1.0']
    
    for animal in animals:
        # Filter data for this animal and drop NaN values
        animal_df = df[df['animal'] == animal].copy()
        animal_df = animal_df.dropna(subset=['outbound_performance'])  # Drop NaN values
        
        # Convert performance values to strings for categorical plotting
        animal_df['outbound_performance'] = animal_df['outbound_performance'].astype(str)
        
        # Ensure performance categories are ordered
        animal_df['outbound_performance'] = pd.Categorical(
            animal_df['outbound_performance'],
            categories=perf_order,
            ordered=True
        )
        
        # Skip this animal if no valid data
        if len(animal_df) == 0:
            print(f"Skipping animal {animal} - no valid data")
            continue
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # First row: Performance distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot performance distributions in first row
        sns.violinplot(data=animal_df, x='outbound_performance', y='tau', ax=ax1,
                      inner=None, palette='viridis', order=perf_order)
        sns.pointplot(data=animal_df, x='outbound_performance', y='tau', ax=ax1,
                     estimator=np.median, color='#21918c',
                     markers='s', markersize=5, linewidth=1,
                     order=perf_order)
        
        ax1.set_title('Neural Timescale by Performance', fontsize=14)
        ax1.set_xlabel('Performance Score', fontsize=12)
        ax1.set_ylabel('Intrinsic Neural Timescale (ms)', fontsize=12)
        ax1.set_ylim(bottom=0)
        
        sns.violinplot(data=animal_df, x='outbound_performance', y='branching_factor', ax=ax2,
                      inner=None, palette='viridis', order=perf_order)
        sns.pointplot(data=animal_df, x='outbound_performance', y='branching_factor', ax=ax2,
                     estimator=np.median, color='#21918c',
                     markers='s', markersize=5, linewidth=1,
                     order=perf_order)
        
        ax2.set_title('Branching Factor by Performance', fontsize=14)
        ax2.set_xlabel('Performance Score', fontsize=12)
        ax2.set_ylabel('Branching Factor', fontsize=12)
        ax2.set_ylim(bottom=0.85, top=1)
        
        # Second row: Area-Performance interactions
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot area-performance interactions in second row
        sns.pointplot(data=animal_df, x='outbound_performance', y='tau', hue='area', 
                     ax=ax3, palette='viridis',
                     markers=['o', 's'], linestyles=['-', '--'],
                     capsize=.1, errwidth=1,
                     estimator=np.mean,
                     order=perf_order)
        
        ax3.set_title('Neural Timescale by Performance and Area', fontsize=14)
        ax3.set_xlabel('Performance Score', fontsize=12)
        ax3.set_ylabel('Intrinsic Neural Timescale (ms)', fontsize=12)
        ax3.legend(title='Area', bbox_to_anchor=(1.05, 1))
        
        sns.pointplot(data=animal_df, x='outbound_performance', y='branching_factor', hue='area',
                     ax=ax4, palette='viridis',
                     markers=['o', 's'], linestyles=['-', '--'],
                     capsize=.1, errwidth=1,
                     estimator=np.mean,
                     order=perf_order)
        
        ax4.set_title('Branching Factor by Performance and Area', fontsize=14)
        ax4.set_xlabel('Performance Score', fontsize=12)
        ax4.set_ylabel('Branching Factor', fontsize=12)
        ax4.legend(title='Area', bbox_to_anchor=(1.05, 1))
        
        # Add overall title for the animal
        plt.suptitle(f'Animal: {animal}', size=30, y=1.05, weight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()