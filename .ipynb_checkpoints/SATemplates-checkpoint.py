import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np

# Colorblind-friendly palette (Paul Tol's bright qualitative scheme)
COLORS = {
    'blue': '#4477AA',
    'cyan': '#66CCEE',
    'green': '#228833',
    'yellow': '#CCBB44',
    'red': '#EE6677',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
}

# Color cycle for multiple series
COLOR_CYCLE = list(COLORS.values())


def check_font_available(font_name):
    """Check if a font is available on the system."""
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    return font_name in available_fonts


def apply_standard_style():
    """Apply standard research style with default sans-serif font."""
    plt.style.use('default')
    
    params = {
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Figure settings
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.prop_cycle': mpl.cycler(color=COLOR_CYCLE),
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.top': True,
        'ytick.right': True,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '0.8',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    }
    
    mpl.rcParams.update(params)
    print("Standard style applied (DejaVu Sans)")


def apply_presentation_style():
    """Apply presentation style with Roboto font (falls back if unavailable)."""
    plt.style.use('default')
    
    # Check if Roboto is available
    if check_font_available('Roboto'):
        font_family = ['Roboto', 'DejaVu Sans', 'Arial']
        print("Presentation style applied (Roboto)")
    else:
        font_family = ['DejaVu Sans', 'Arial', 'Helvetica']
        print("Presentation style applied (Roboto not found, using DejaVu Sans)")
    
    params = {
        # Font settings - larger for presentations
        'font.family': 'sans-serif',
        'font.sans-serif': font_family,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        
        # Figure settings - larger for presentations
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes settings - thicker lines for visibility
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.prop_cycle': mpl.cycler(color=COLOR_CYCLE),
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.top': True,
        'ytick.right': True,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '0.8',
        
        # Line settings - thicker for presentations
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
    }
    
    mpl.rcParams.update(params)


def create_scatter_plot(x, y, labels=None, xlabel='X', ylabel='Y', title='', 
                       colors=None, markers=None, alpha=0.7, s=50):
    """
    Create a scatter plot with template styling.
    
    Parameters:
    -----------
    x, y : array-like or list of array-like
        Data for x and y axes. Can be single arrays or lists of arrays for multiple series.
    labels : str or list of str, optional
        Legend labels for each series.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    colors : str or list, optional
        Colors for each series. Uses template colors if None.
    markers : str or list, optional
        Marker styles for each series. Default is 'o'.
    alpha : float
        Transparency of markers.
    s : float or array-like
        Marker size.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots()
    
    # Handle single series or multiple series
    if not isinstance(x, list):
        x, y = [x], [y]
        labels = [labels] if labels else [None]
    
    if colors is None:
        colors = COLOR_CYCLE[:len(x)]
    elif not isinstance(colors, list):
        colors = [colors]
    
    if markers is None:
        markers = ['o'] * len(x)
    elif not isinstance(markers, list):
        markers = [markers]
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = labels[i] if i < len(labels) else None
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.scatter(xi, yi, label=label, color=color, marker=marker, 
                  alpha=alpha, s=s, edgecolors='none')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(labels):
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


def create_line_plot(x, y, labels=None, xlabel='X', ylabel='Y', title='',
                    colors=None, linestyles=None, markers=None):
    """
    Create a line plot with template styling.
    
    Parameters:
    -----------
    x, y : array-like or list of array-like
        Data for x and y axes. Can be single arrays or lists of arrays for multiple series.
    labels : str or list of str, optional
        Legend labels for each series.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    colors : str or list, optional
        Colors for each series. Uses template colors if None.
    linestyles : str or list, optional
        Line styles. Default is '-'.
    markers : str or list, optional
        Marker styles. Default is None.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots()
    
    # Handle single series or multiple series
    if not isinstance(x, list):
        x, y = [x], [y]
        labels = [labels] if labels else [None]
    
    if colors is None:
        colors = COLOR_CYCLE[:len(x)]
    elif not isinstance(colors, list):
        colors = [colors]
    
    if linestyles is None:
        linestyles = ['-'] * len(x)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles]
    
    if markers is not None and not isinstance(markers, list):
        markers = [markers]
    
    for i, (xi, yi) in enumerate(zip(x, y)):
        label = labels[i] if i < len(labels) else None
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)] if markers else None
        
        ax.plot(xi, yi, label=label, color=color, linestyle=linestyle, 
               marker=marker, markevery=max(1, len(xi)//10))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(labels):
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


def create_benchmark_plot(predicted, actual, xlabel='Actual', ylabel='Predicted',
                         title='Benchmark Comparison', show_identity=True,
                         show_stats=True, show_error_hist=True, hist_position='right'):
    """
    Create a 45-degree scatter plot for benchmarking predictions vs actual values.
    
    Parameters:
    -----------
    predicted : array-like
        Predicted values.
    actual : array-like
        Actual values.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    show_identity : bool
        Whether to show the 45-degree identity line.
    show_stats : bool
        Whether to display R² and RMSE statistics.
    show_error_hist : bool
        Whether to show histogram of errors on a secondary axis.
    hist_position : str
        Position of histogram: 'right' or 'top'.
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    errors = predicted - actual
    
    # Create main figure and axis
    fig, ax_main = plt.subplots(figsize=(7, 6))
    
    # Main scatter plot
    ax_main.scatter(actual, predicted, color=COLORS['blue'], alpha=0.6, 
                   s=50, edgecolors='none', label='Data', zorder=3)
    
    # 45-degree line
    if show_identity:
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax_main.plot([min_val, max_val], [min_val, max_val], 
                    'k--', linewidth=1.5, alpha=0.7, label='Perfect prediction', zorder=2)
    
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.set_aspect('equal', adjustable='box')
    
    # Add error histogram on secondary axis
    ax_hist = None
    if show_error_hist:
        if hist_position == 'right':
            # Create secondary y-axis on the right
            ax_hist = ax_main.twinx()
            ax_hist.hist(errors, bins=20, orientation='horizontal', 
                        color=COLORS['red'], alpha=0.5, edgecolor='black', 
                        linewidth=0.5, zorder=1)
            ax_hist.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
            ax_hist.set_ylabel('Error (Predicted - Actual)', color=COLORS['red'])
            ax_hist.tick_params(axis='y', labelcolor=COLORS['red'])
            # Match error axis limits to main plot y-axis
            ax_hist.set_ylim(ax_main.get_ylim())
        else:  # top
            # Create secondary x-axis on top
            ax_hist = ax_main.twiny()
            ax_hist.hist(errors, bins=20, orientation='vertical',
                        color=COLORS['red'], alpha=0.5, edgecolor='black', 
                        linewidth=0.5, zorder=1)
            ax_hist.axvline(x=0, color='k', linestyle=':', linewidth=1, alpha=0.5, zorder=2)
            ax_hist.set_xlabel('Error (Predicted - Actual)', color=COLORS['red'])
            ax_hist.tick_params(axis='x', labelcolor=COLORS['red'])
            # Match error axis limits to main plot x-axis
            ax_hist.set_xlim(ax_main.get_xlim())
    
    # Calculate and display statistics
    if show_stats:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted)
        rmse = np.sqrt(np.mean((predicted - actual)**2))
        
        stats_text = f'R² = {r_value**2:.3f}\nRMSE = {rmse:.3f}'
        ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.9, edgecolor='0.8'), zorder=4)
    
    ax_main.legend(loc='lower right')
    plt.tight_layout()
    
    if show_error_hist:
        return fig, (ax_main, ax_hist)
    else:
        return fig, ax_main


def create_bar_plot(categories, values, labels=None, xlabel='', ylabel='Value',
                   title='', colors=None, orientation='vertical'):
    """
    Create a bar plot with template styling.
    
    Parameters:
    -----------
    categories : array-like
        Category names or x-positions.
    values : array-like or list of array-like
        Bar heights. Can be single array or list of arrays for grouped bars.
    labels : str or list of str, optional
        Legend labels for each group.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    colors : str or list, optional
        Colors for each group. Uses template colors if None.
    orientation : str
        'vertical' or 'horizontal'.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots()
    
    # Handle single group or multiple groups
    if not isinstance(values[0], (list, np.ndarray)) or len(np.array(values).shape) == 1:
        values = [values]
        labels = [labels] if labels else [None]
    
    if colors is None:
        colors = COLOR_CYCLE[:len(values)]
    elif not isinstance(colors, list):
        colors = [colors]
    
    n_groups = len(values)
    n_cats = len(categories)
    bar_width = 0.8 / n_groups
    
    x = np.arange(n_cats)
    
    for i, vals in enumerate(values):
        offset = (i - n_groups/2 + 0.5) * bar_width
        label = labels[i] if i < len(labels) else None
        color = colors[i % len(colors)]
        
        if orientation == 'vertical':
            ax.bar(x + offset, vals, bar_width, label=label, color=color)
        else:
            ax.barh(x + offset, vals, bar_width, label=label, color=color)
    
    if orientation == 'vertical':
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    
    if title:
        ax.set_title(title)
    if any(labels):
        ax.legend()
    
    plt.tight_layout()
    return fig, ax

def create_bar_plot(categories, values, xlabel='Categories', ylabel='Values',
                   title='', labels=None, horizontal=False, group_labels=None):
    """
    Create a bar plot with single or grouped bars.
    
    Parameters:
    -----------
    categories : array-like
        Category labels.
    values : array-like or list of array-like
        Values for each category. Can be 2D for grouped bars.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    labels : str or list of str
        Labels for legend (for grouped bars).
    horizontal : bool
        If True, create horizontal bar plot.
    group_labels : list of str
        Alternative to categories for grouped bars.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy arrays
    values = np.array(values)
    
    # Check if grouped bars
    if values.ndim == 2:
        n_groups = len(categories)
        n_bars = values.shape[0]
        bar_width = 0.8 / n_bars
        x = np.arange(n_groups)
        
        for i in range(n_bars):
            offset = (i - n_bars/2 + 0.5) * bar_width
            if horizontal:
                ax.barh(x + offset, values[i], bar_width, 
                       color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                       label=labels[i] if labels else None)
            else:
                ax.bar(x + offset, values[i], bar_width,
                      color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
                      label=labels[i] if labels else None)
        
        if horizontal:
            ax.set_yticks(x)
            ax.set_yticklabels(group_labels if group_labels else categories)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels if group_labels else categories)
    else:
        # Single bar plot
        if horizontal:
            ax.barh(categories, values, color=COLORS['blue'],
                   label=labels if isinstance(labels, str) else None)
        else:
            ax.bar(categories, values, color=COLORS['blue'],
                  label=labels if isinstance(labels, str) else None)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    if labels:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def create_fit_plot(x, y, fitparams, xlabel='X', ylabel='Y', title='',
                   data_label='Data', fit_label='Fit', show_equation=True,
                   show_stats=True, n_points=200):
    """
    Create a plot with data points and a fitted polynomial curve.
    
    Parameters:
    -----------
    x, y : array-like
        Data points to plot.
    fitparams : array-like
        Polynomial coefficients in increasing order of power.
        E.g., [c0, c1, c2] represents: y = c0 + c1*x + c2*x^2
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    data_label : str
        Label for data points.
    fit_label : str
        Label for fitted curve.
    show_equation : bool
        Whether to display the fit equation on the plot.
    show_stats : bool
        Whether to display R² statistic.
    n_points : int
        Number of points to use for plotting the smooth fit curve.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    fitparams = np.array(fitparams)
    
    # Determine polynomial order
    fitorder = len(fitparams) - 1
    
    # Plot data points
    ax.scatter(x, y, color=COLORS['blue'], s=50, alpha=0.6,
              label=data_label, zorder=3)
    
    # Generate smooth curve for fitted line
    x_fit = np.linspace(x.min(), x.max(), n_points)
    y_fit = np.polyval(fitparams[::-1], x_fit)  # polyval expects decreasing order
    
    # Plot fitted curve
    ax.plot(x_fit, y_fit, color=COLORS['red'], linewidth=2,
           label=fit_label, zorder=2)
    
    # Calculate R²
    if show_stats:
        y_pred = np.polyval(fitparams[::-1], x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    
    # Create equation string
    if show_equation:
        eq_parts = []
        for i, coef in enumerate(fitparams):
            if abs(coef) < 1e-10:  # Skip very small coefficients
                continue
            
            # Format coefficient
            coef_str = f"{coef:.3g}"
            
            if i == 0:
                eq_parts.append(coef_str)
            elif i == 1:
                eq_parts.append(f"{coef_str}x")
            else:
                eq_parts.append(f"{coef_str}x^{{{i}}}")
        
        equation = " + ".join(eq_parts).replace("+ -", "- ")
        
        # Create text box
        textstr = f'$y = {equation}'
        if show_stats:
            textstr += f'\n$R^2 = {r_squared:.4f}$'
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment='top', bbox=props, fontsize=10)
    elif show_stats:
        # Show only R² if equation is hidden
        textstr = f'$R^2 = {r_squared:.4f}$'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment='top', bbox=props, fontsize=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax



# Example usage
if __name__ == '__main__':
    # Apply standard style for research papers
    apply_standard_style()
    
    # Generate sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 50)
    
    # Line plot example
    fig1, ax1 = create_line_plot(
        [x, x], [y1, y2],
        labels=['Sin wave', 'Cos wave'],
        xlabel='Time (s)',
        ylabel='Amplitude',
        title='Example Line Plot'
    )
    
    # Scatter plot example
    fig2, ax2 = create_scatter_plot(
        [x, x], [y1, y2],
        labels=['Data A', 'Data B'],
        xlabel='X variable',
        ylabel='Y variable',
        title='Example Scatter Plot'
    )
    
    # Benchmark plot example
    actual = np.random.rand(100) * 100
    predicted = actual + np.random.normal(0, 10, 100)
    fig3, ax3_main = create_benchmark_plot(  #, ax3_hist)
        predicted, actual,
        xlabel='Actual Values',
        ylabel='Predicted Values',
        title='Prediction Benchmark',
        show_error_hist=False
#        hist_position= 'top'#'right'  # or 'top'
    )
    
    # Bar plot example
    categories = ['Method A', 'Method B', 'Method C', 'Method D']
    values1 = [23, 45, 56, 32]
    values2 = [18, 38, 48, 28]
    fig4, ax4 = create_bar_plot(
        categories, [values1, values2],
        labels=['Dataset 1', 'Dataset 2'],
        ylabel='Performance Score',
        title='Example Bar Plot'
    )
    
    plt.show()