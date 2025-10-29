"""
Advanced Visualizations for One NZ App Review Analysis
Interactive Dashboards and Publication-Quality Plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Color palettes
SENTIMENT_COLORS = {
    'Positive': '#2ecc71',
    'Neutral': '#f39c12',
    'Negative': '#e74c3c'
}


# ============================================================================
# VISUALIZATION 1: EXECUTIVE DASHBOARD OVERVIEW
# ============================================================================

def create_executive_dashboard(df):
    """
    Create comprehensive executive dashboard
    """
    print("\nCreating Executive Dashboard...")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Overall Sentiment Distribution',
            'Review Volume Over Time',
            'Average Rating Trend',
            'Top Issues Mentioned',
            'Response Rate Impact',
            'Negative Sentiment by App Version'
        ),
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Sentiment Distribution Pie Chart
    sentiment_counts = df['sentiment_label'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=[SENTIMENT_COLORS[s] for s in sentiment_counts.index]),
            hole=0.4
        ),
        row=1, col=1
    )

    # 2. Review Volume Over Time
    daily_reviews = df.groupby('review_date').size().reset_index(name='count')
    fig.add_trace(
        go.Scatter(
            x=daily_reviews['review_date'],
            y=daily_reviews['count'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#3498db', width=2)
        ),
        row=1, col=2
    )

    # 3. Average Rating Trend
    df['review_date'] = pd.to_datetime(df['review_date'])
    monthly_avg = df.groupby(df['review_date'].apply(lambda x: x.strftime('%Y-%m')))['score'].mean()
    fig.add_trace(
        go.Scatter(
            x=monthly_avg.index,
            y=monthly_avg.values,
            mode='lines+markers',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )

    # 4. Top Issues Bar Chart
    text_column = 'content_clean_nostop' if 'content_clean_nostop' in df.columns else 'content_clean'
    negative_reviews = df[df['sentiment_label'] == 'Negative'][text_column]

    issue_keywords = {
        # Authentication & Access
        'Login/Authentication Issues': [
            'login', 'log in', 'sign in', 'signin', 'cant log', "can't log",
            'password', 'locked out', 'authentication', 'verify', 'verification',
            'forgot password', 'reset password', 'account access'
        ],

        # Performance & Stability
        'App Crashes/Freezing': [
            'crash', 'crashes', 'freeze', 'frozen', 'hang', 'stuck',
            'loading forever', 'stops working', 'force close', 'unresponsive',
            'black screen', 'white screen', 'not responding'
        ],

        'Slow Performance': [
            'slow', 'lag', 'laggy', 'sluggish', 'takes forever',
            'speed', 'performance', 'loading time', 'wait', 'waiting'
        ],

        # Billing & Payment
        'Billing/Charging Issues': [
            'charge', 'charged', 'bill', 'billing', 'payment', 'money',
            'fee', 'cost', 'expensive', 'overcharge', 'wrong charge',
            'unexpected charge', 'double charge', 'incorrect bill'
        ],

        'Auto Top-up Problems': [
            'auto topup', 'auto top up', 'automatic topup', 'auto payment',
            'keep charging', 'stop charging', 'deducted', 'taken money'
        ],

        # Data & Usage
        'Data Usage Issues': [
            'data', 'usage', 'data usage', 'using data', 'data gone',
            'data disappear', 'data vanish', 'eating data', 'data drain'
        ],

        'Balance/Credit Issues': [
            'balance', 'credit', 'top up', 'topup', 'recharge', 'reload',
            'cant see balance', 'balance missing', 'credit missing'
        ],

        # Functionality
        'Missing Features': [
            'missing', 'removed', 'gone', 'disappeared', 'unavailable',
            'cant find', "can't find", 'where is', 'no longer', 'taken away'
        ],

        'Feature Not Working': [
            'not working', 'doesnt work', "doesn't work", 'broken', 'error',
            'failed', 'fails', 'wont work', "won't work", 'cant use'
        ],

        # UI/UX
        'User Interface Problems': [
            'interface', 'ui', 'design', 'layout', 'confusing', 'complicated',
            'hard to use', 'difficult', 'cant navigate', 'poor design'
        ],

        # Updates & Versions
        'Update Problems': [
            'update', 'updated', 'new version', 'latest version', 'upgrade',
            'after update', 'since update', 'broken after', 'worse after'
        ],

        # Customer Service
        'Customer Service Issues': [
            'support', 'service', 'help', 'customer service', 'no help',
            'response', 'reply', 'contact', 'cant contact', 'no response',
            'unhelpful', 'rude', 'poor service'
        ],

        # Network & Connectivity
        'Network/Connection Issues': [
            'network', 'connection', 'signal', 'coverage', 'internet',
            'wifi', 'connect', 'disconnect', 'no service', 'poor signal'
        ],

        # Installation & Compatibility
        'Installation/Compatibility': [
            'install', 'download', 'wont install', 'cant install',
            'compatible', 'compatibility', 'phone', 'device', 'version'
        ],

        # Account Management
        'Account Issues': [
            'account', 'profile', 'settings', 'cant change', 'locked',
            'suspended', 'closed', 'cant access account'
        ]
    }

    issue_counts = {}
    for issue, keywords in issue_keywords.items():
        count = negative_reviews.apply(
            lambda x: any(k in str(x).lower() for k in keywords)
        ).sum()
        issue_counts[issue] = count

    fig.add_trace(
        go.Bar(
            x=list(issue_counts.keys()),
            y=list(issue_counts.values()),
            marker=dict(color='#e74c3c')
        ),
        row=2, col=2
    )

    # 5. Response Rate Impact
    #response_impact = df.groupby('has_reply')['score'].mean()
    response_impact = df['has_reply'].value_counts()
    fig.add_trace(
        go.Bar(
            x=['No Reply', 'Has Reply'],
            y=response_impact.values,
            marker=dict(color=['#95a5a6', '#27ae60'])
        ),
        row=3, col=1
    )

    # 6. Sentiment by Version
    version_sentiment = df.groupby('major_version')['roberta_sentiment'].apply(
        lambda x: (x == 'Negative').sum()
    ).sort_values(ascending=False).head(8)

    fig.add_trace(
        go.Bar(
            x=version_sentiment.index,
            y=version_sentiment.values,
            marker=dict(color='#e67e22')
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="One NZ App Review Analysis - Executive Dashboard",
        title_font_size=20,
        showlegend=False,
        height=1200,
        template='plotly_white'
    )

    fig.write_html('executive_dashboard.html')
    print("Dashboard saved as 'executive_dashboard.html'")

    return fig


# ============================================================================
# VISUALIZATION 2: SENTIMENT TRENDS ANALYSIS
# ============================================================================

def create_sentiment_trends(df):
    """
    Create detailed sentiment trend visualizations
    """
    print("\nCreating Sentiment Trends...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Sentiment over time (stacked area)
    monthly_sentiment = pd.crosstab(
        df['review_date'].apply(lambda x: x.strftime('%Y-%m')),
        df['sentiment_label']
    )

    monthly_sentiment.plot(
        kind='area',
        stacked=True,
        color=[SENTIMENT_COLORS['Negative'], SENTIMENT_COLORS['Neutral'],
               SENTIMENT_COLORS['Positive']],
        alpha=0.7,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Sentiment Distribution Over Time (Stacked)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Number of Reviews')
    axes[0, 0].legend(title='Sentiment')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Sentiment percentage over time
    monthly_pct = pd.crosstab(
        df['review_date'].apply(lambda x: x.strftime('%Y-%m')),
        df['sentiment_label'],
        normalize='index'
    ) * 100

    monthly_pct.plot(
        kind='line',
        color=[SENTIMENT_COLORS['Negative'], SENTIMENT_COLORS['Neutral'],
               SENTIMENT_COLORS['Positive']],
        linewidth=3,
        marker='o',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Sentiment Percentage Trends',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].legend(title='Sentiment')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score distribution by sentiment
    sns.violinplot(
        data=df,
        x='sentiment_label',
        y='score',
        palette=SENTIMENT_COLORS,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Score Distribution by Sentiment Category',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sentiment')
    axes[1, 0].set_ylabel('Rating Score')

    # 4. Review count by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sentiment = pd.crosstab(df['day_of_week'], df['sentiment_label'])
    day_sentiment = day_sentiment.reindex(day_order)

    day_sentiment.plot(
        kind='bar',
        stacked=True,
        color=[SENTIMENT_COLORS['Negative'], SENTIMENT_COLORS['Neutral'],
               SENTIMENT_COLORS['Positive']],
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Reviews by Day of Week',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Number of Reviews')
    axes[1, 1].legend(title='Sentiment')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('sentiment_trends.png', dpi=300, bbox_inches='tight')
    print("Saved as 'sentiment_trends.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 3: WORD CLOUDS
# ============================================================================

def create_wordclouds(df):
    """
    Create word clouds for different sentiment categories using stopword-removed text
    """
    print("\nCreating Word Clouds...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sentiments = ['Positive', 'Neutral', 'Negative']

    # Use stopword-removed text for cleaner word clouds
    text_column = 'content_clean_nostop' if 'content_clean_nostop' in df.columns else 'content_clean'

    for idx, sentiment in enumerate(sentiments):
        text = ' '.join(df[df['sentiment_label'] == sentiment][text_column].dropna())

        # Additional custom stopwords specific to app reviews
        custom_stopwords = set([
            'app', 'one', 'nz', 'vodafone', 'phone', 'mobile',
            'use', 'get', 'would', 'could', 'time', 'now', 'just',
            'even', 'also', 'back', 'still', 'much', 'well', 'like'
        ])

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            #colormap='RdYlGn' if sentiment == 'Positive' else 'RdYlGn_r',
            colormap='RdYlGn',
            max_words=100,
            relative_scaling=0.5,
            stopwords=custom_stopwords,
            min_font_size=10,
            collocations=False  # Avoid duplicate word combinations
        ).generate(text)

        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{sentiment} Reviews',
                            fontsize=16, fontweight='bold',
                            color=SENTIMENT_COLORS[sentiment])

    plt.tight_layout()
    plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
    print("Saved as 'wordclouds.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 4: MODEL COMPARISON
# ============================================================================

def create_model_comparison(results):
    """
    Visualize ML model performance comparison
    """
    print("\nCreating Model Comparison Chart...")

    # Extract metrics
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_names))
    width = 0.2

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        ax.bar(x + idx * width, values, width,
               label=metric.replace('_', ' ').title(),
               color=colors[idx], alpha=0.8)

    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ML Model Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved as 'model_comparison.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 5: CONFUSION MATRIX HEATMAP
# ============================================================================

def create_confusion_matrix(y_test, y_pred, model_name):
    """
    Create confusion matrix heatmap
    """
    from sklearn.metrics import confusion_matrix

    print(f"\nCreating Confusion Matrix for {model_name}...")

    cm = confusion_matrix(y_test, y_pred,
                          labels=['Negative', 'Neutral', 'Positive'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix - {model_name}',
              fontsize=14, fontweight='bold')
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved as 'confusion_matrix.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 6: INTERACTIVE TIMELINE WITH LAST YEAR ANALYSIS
# ============================================================================

def create_interactive_timeline(df):
    """
    Create interactive timeline with sentiment analysis and last year deep dive
    """
    print("\nCreating Interactive Timeline with Last Year Analysis...")

    # Get last 1 year of data
    latest_date = df['review_date'].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)
    df_last_year = df[df['review_date'] >= one_year_ago].copy()

    print(f"  Full dataset: {len(df)} reviews")
    print(f"  Last year ({one_year_ago.date()} to {latest_date}): {len(df_last_year)} reviews")

    # Create subplots for comprehensive timeline
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Review Volume and Sentiment Over Time (Full Period)',
            'Last Year: App Version vs Average Rating',
            'Last Year: Review Distribution by App Version',
            'Last Year: Top Issues by App Version'
        ),
        vertical_spacing=0.08,
        specs=[
            [{"type": "scatter"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "bar"}]
        ],
        row_heights=[0.3, 0.25, 0.2, 0.25]
    )

    # ========================================================================
    # CHART 1: Full Timeline - Sentiment Over Time
    # ========================================================================


    daily_data = df.groupby(['review_date', 'sentiment_label']).size().reset_index(name='count')

    for sentiment in ['Positive', 'Neutral', 'Negative']:
        sentiment_data = daily_data[daily_data['sentiment_label'] == sentiment]
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['review_date'],
                y=sentiment_data['count'],
                mode='lines',
                name=sentiment,
                line=dict(color=SENTIMENT_COLORS[sentiment], width=2),
                legendgroup='sentiment',
                showlegend=True
            ),
            row=1, col=1
        )

    # Add vertical line for "last year" marker
    # fig.add_vline(
    #     x=one_year_ago,
    #     line_dash="dash",
    #     line_color="gray",
    #     annotation_text="Last Year",
    #     row=1, col=1
    # )

    # Convert to datetime string format for plotly compatibility

    fig.add_shape(
        type="line",
        x0=one_year_ago,
        x1=one_year_ago,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash"),
        row=1, col=1
    )

    # Add annotation for the line
    fig.add_annotation(
        x=one_year_ago,
        y=1,
        yref="paper",
        text="Last Year",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color="gray"),
        row=1, col=1
    )

    # ========================================================================
    # CHART 2: Last Year - App Version vs Rating
    # ========================================================================
    #version_rating_last_year = df_last_year.groupby('major_version').agg({
    version_rating_last_year = df_last_year.groupby('appVersion').agg({
        'score': 'mean',
        'reviewId': 'count'
    }).reset_index()
    version_rating_last_year.columns = ['appVersion', 'avg_rating', 'review_count']

    # Filter versions with at least 10 reviews
    version_rating_last_year = version_rating_last_year[
        version_rating_last_year['review_count'] >= 10
        ].sort_values('appVersion', ascending = True)
    #sort_values('avg_rating', ascending=False)

    # Color code by rating
    colors = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71'
              for x in version_rating_last_year['avg_rating']]

    fig.add_trace(
        go.Bar(
            x=version_rating_last_year['appVersion'],
            y=version_rating_last_year['avg_rating'],
            marker=dict(color=colors),
            text=version_rating_last_year['avg_rating'].round(2),
            textposition='outside',
            name='Avg Rating',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Rating: %{y:.2f}<br>Reviews: %{customdata}<extra></extra>',
            customdata=version_rating_last_year['review_count']
        ),
        row=2, col=1
    )

    # Add reference lines
    fig.add_hline(y=3, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=4, line_dash="dash", line_color="orange", opacity=0.5, row=2, col=1)

    # ========================================================================
    # CHART 3: Last Year - Review Distribution by Version
    # ========================================================================
    version_sentiment_last_year = pd.crosstab(
        df_last_year['appVersion'],
        df_last_year['sentiment_label'],
        normalize='index'
    ).reset_index()

    # Filter to top versions
    top_versions_last_year = df_last_year['appVersion'].value_counts().head(8).index
    version_sentiment_last_year = version_sentiment_last_year[
        version_sentiment_last_year['appVersion'].isin(top_versions_last_year)
    ]
    version_sentiment_last_year = version_sentiment_last_year.sort_values(
        by='appVersion',
        ascending=True
    )
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        if sentiment in version_sentiment_last_year.columns:
            fig.add_trace(
                go.Bar(
                    x=version_sentiment_last_year['appVersion'],
                    y=version_sentiment_last_year[sentiment] * 100,
                    name=sentiment,
                    marker=dict(color=SENTIMENT_COLORS[sentiment]),
                    legendgroup='sentiment2',
                    showlegend=False
                ),
                row=3, col=1
            )

    # ========================================================================
    # CHART 4: Last Year - Top Issues by App Version
    # ========================================================================

    # Get text column
    text_column = 'content_clean_nostop' if 'content_clean_nostop' in df.columns else 'content_clean'

    # # Define top issues to track
    # issue_keywords = {
    #     'Login': ['login', 'log in', 'password'],
    #     'Crash': ['crash', 'freeze', 'hang'],
    #     'Billing': ['charge', 'bill', 'payment'],
    #     'Data': ['data', 'usage'],
    #     'Slow': ['slow', 'lag']
    # }
    issue_keywords = {
        # Authentication & Access
        'Login/Authentication Issues': [
            'login', 'log in', 'sign in', 'signin', 'cant log', "can't log",
            'password', 'locked out', 'authentication', 'verify', 'verification',
            'forgot password', 'reset password', 'account access'
        ],

        # Performance & Stability
        'App Crashes/Freezing': [
            'crash', 'crashes', 'freeze', 'frozen', 'hang', 'stuck',
            'loading forever', 'stops working', 'force close', 'unresponsive',
            'black screen', 'white screen', 'not responding'
        ],

        'Slow Performance': [
            'slow', 'lag', 'laggy', 'sluggish', 'takes forever',
            'speed', 'performance', 'loading time', 'wait', 'waiting'
        ],

        # Billing & Payment
        'Billing/Charging Issues': [
            'charge', 'charged', 'bill', 'billing', 'payment', 'money',
            'fee', 'cost', 'expensive', 'overcharge', 'wrong charge',
            'unexpected charge', 'double charge', 'incorrect bill'
        ],

        'Auto Top-up Problems': [
            'auto topup', 'auto top up', 'automatic topup', 'auto payment',
            'keep charging', 'stop charging', 'deducted', 'taken money'
        ],

        # Data & Usage
        'Data Usage Issues': [
            'data', 'usage', 'data usage', 'using data', 'data gone',
            'data disappear', 'data vanish', 'eating data', 'data drain'
        ],

        'Balance/Credit Issues': [
            'balance', 'credit', 'top up', 'topup', 'recharge', 'reload',
            'cant see balance', 'balance missing', 'credit missing'
        ],

        # Functionality
        'Missing Features': [
            'missing', 'removed', 'gone', 'disappeared', 'unavailable',
            'cant find', "can't find", 'where is', 'no longer', 'taken away'
        ],

        'Feature Not Working': [
            'not working', 'doesnt work', "doesn't work", 'broken', 'error',
            'failed', 'fails', 'wont work', "won't work", 'cant use'
        ],

        # UI/UX
        'User Interface Problems': [
            'interface', 'ui', 'design', 'layout', 'confusing', 'complicated',
            'hard to use', 'difficult', 'cant navigate', 'poor design'
        ],

        # Updates & Versions
        'Update Problems': [
            'update', 'updated', 'new version', 'latest version', 'upgrade',
            'after update', 'since update', 'broken after', 'worse after'
        ],

        # Customer Service
        'Customer Service Issues': [
            'support', 'service', 'help', 'customer service', 'no help',
            'response', 'reply', 'contact', 'cant contact', 'no response',
            'unhelpful', 'rude', 'poor service'
        ],

        # Network & Connectivity
        'Network/Connection Issues': [
            'network', 'connection', 'signal', 'coverage', 'internet',
            'wifi', 'connect', 'disconnect', 'no service', 'poor signal'
        ],

        # Installation & Compatibility
        'Installation/Compatibility': [
            'install', 'download', 'wont install', 'cant install',
            'compatible', 'compatibility', 'phone', 'device', 'version'
        ],

        # Account Management
        'Account Issues': [
            'account', 'profile', 'settings', 'cant change', 'locked',
            'suspended', 'closed', 'cant access account'
        ]
    }

    # Calculate issue frequency by version (last year, negative reviews only)
    version_issues = []

    for version in top_versions_last_year:
        version_data = df_last_year[
            (df_last_year['appVersion'] == version) &
            (df_last_year['sentiment_label'] == 'Negative')
            ]

        if len(version_data) > 0:
            for issue, keywords in issue_keywords.items():
                count = version_data[text_column].apply(
                    lambda x: any(k in str(x).lower() for k in keywords) if pd.notna(x) else False
                ).sum()

                percentage = (count / len(version_data)) * 100

                version_issues.append({
                    'version': version,
                    'issue': issue,
                    'percentage': percentage,
                    'count': count
                })

    version_issues_df = pd.DataFrame(version_issues)

    # Plot top issue per version
    if len(version_issues_df) > 0:
        # Get the dominant issue for each version
        dominant_issues = version_issues_df.loc[
            version_issues_df.groupby('version')['percentage'].idxmax()
        ]
        dominant_issues = dominant_issues.sort_values(
            by='version',
            ascending=True
        )
        # Create color map for issues
        issue_colors = {
            'Login/Authentication Issues': '#0d47a1',  # Dark blue - high severity
            'App Crashes/Freezing': '#1565c0',  # Very dark blue - high severity
            'Slow Performance': '#1976d2',  # Dark blue - medium-high severity
            'Billing/Charging Issues': '#1e88e5',  # Medium-dark blue - high severity
            'Auto Top-up Problems': '#2196f3',  # Medium blue - medium severity
            'Data Usage Issues': '#42a5f5',  # Medium-light blue - medium severity
            'Balance/Credit Issues': '#64b5f6',  # Light-medium blue - medium severity
            'Missing Features': '#90caf9',  # Light blue - lower severity
            'Feature Not Working': '#1976d2',  # Dark blue - medium-high severity
            'User Interface Problems': '#bbdefb',  # Very light blue - lower severity
            'Update Problems': '#2196f3',  # Medium blue - medium severity
            'Customer Service Issues': '#1565c0',  # Very dark blue - high severity
            'Network/Connection Issues': '#1e88e5',  # Medium-dark blue - high severity
            'Installation/Compatibility': '#64b5f6',  # Light-medium blue - medium severity
            'Account Issues': '#0d47a1'  # Dark blue - high severity
        }

        colors = [issue_colors.get(issue, '#3498db') for issue in dominant_issues['issue']]

        fig.add_trace(
            go.Bar(
                x=dominant_issues['version'],
                y=dominant_issues['percentage'],
                marker=dict(color=colors),
                text=dominant_issues['issue'],
                textposition='outside',
                name='Top Issue',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Issue: %{text}<br>Frequency: %{y:.1f}%<br>Count: %{customdata}<extra></extra>',
                customdata=dominant_issues['count']
            ),
            row=4, col=1
        )

    # ========================================================================
    # Update Layout
    # ========================================================================
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Review Count", row=1, col=1)

    fig.update_xaxes(title_text="App Version", row=2, col=1)
    fig.update_yaxes(title_text="Average Rating", range=[0, 5], row=2, col=1)

    fig.update_xaxes(title_text="App Version", row=3, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=3, col=1)

    fig.update_xaxes(title_text="App Version", row=4, col=1)
    fig.update_yaxes(title_text="Issue Frequency (%)", row=4, col=1)

    fig.update_layout(
        title_text="One NZ App Analysis: Interactive Timeline with Last Year Deep Dive",
        title_font_size=18,
        height=1600,
        template='plotly_white',
        hovermode='x unified',
        barmode='stack',
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.write_html('interactive_timeline.html')
    print("Saved as 'interactive_timeline.html'")

    # ========================================================================
    # Generate Summary Statistics
    # ========================================================================
    print("\n" + "=" * 60)
    print("LAST YEAR SUMMARY (App Version Analysis)")
    print("=" * 60)

    print(f"\nPeriod: {one_year_ago.date()} to {latest_date}")
    print(f"Total Reviews: {len(df_last_year):,}")

    print("\nTop 5 App Versions by Review Count:")
    print(df_last_year['appVersion'].value_counts().head(5))

    print("\nApp Version Ratings (Last Year):")
    print(version_rating_last_year[['appVersion', 'avg_rating', 'review_count']].to_string(index=False))

    if len(version_issues_df) > 0:
        print("\nTop Issues by Version (% of Negative Reviews):")
        pivot_issues = version_issues_df.pivot(
            index='version',
            columns='issue',
            values='percentage'
        ).fillna(0).round(1)
        print(pivot_issues)

    print("=" * 60)

    return fig

# ============================================================================
# VISUALIZATION 7: APP VERSION ANALYSIS
# ============================================================================

def create_version_analysis(df):
    """
    Analyze sentiment across app versions
    """
    print("\nCreating App Version Analysis...")

    # Get top versions
    top_versions = df['major_version'].value_counts().head(8).index
    df_versions = df[df['major_version'].isin(top_versions)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Average rating by version
    version_rating = df_versions.groupby('major_version')['score'].mean().sort_values()

    colors = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71'
              for x in version_rating.values]

    version_rating.plot(kind='barh', color=colors, ax=axes[0])
    axes[0].set_title('Average Rating by App Version',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Average Rating')
    axes[0].set_ylabel('App Version')
    axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Critical')
    axes[0].axvline(x=4, color='orange', linestyle='--', alpha=0.5, label='Warning')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')

    # 2. Sentiment distribution by version
    version_sentiment = pd.crosstab(
        df_versions['major_version'],
        df_versions['sentiment_label'],
        normalize='index'
    ) * 100

    version_sentiment.plot(
        kind='barh',
        stacked=True,
        color=[SENTIMENT_COLORS['Negative'], SENTIMENT_COLORS['Neutral'],
               SENTIMENT_COLORS['Positive']],
        ax=axes[1]
    )
    axes[1].set_title('Sentiment Distribution by App Version (%)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Percentage')
    axes[1].set_ylabel('App Version')
    axes[1].legend(title='Sentiment', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig('version_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved as 'version_analysis.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 8: RESPONSE IMPACT ANALYSIS
# ============================================================================

def create_response_analysis(df):
    """
    Analyze the impact of company responses
    """
    print("\nCreating Response Impact Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Response rate over time - FIXED X-AXIS OVERLAP
    monthly_response = df.groupby(
        df['review_date'].apply(lambda x: x.strftime('%Y-%m'))
    )['has_reply'].mean() * 100

    axes[0, 0].plot(monthly_response.index, monthly_response.values,
                    marker='o', linewidth=2, markersize=6, color='#3498db')
    axes[0, 0].fill_between(range(len(monthly_response)), monthly_response.values,
                            alpha=0.3, color='#3498db')
    axes[0, 0].set_title('Response Rate Over Time',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Response Rate (%)')

    # Fix overlapping x-axis labels
    axes[0, 0].set_xticks(range(0, len(monthly_response), max(1, len(monthly_response) // 6)))
    axes[0, 0].set_xticklabels(
        [monthly_response.index[i] for i in range(0, len(monthly_response), max(1, len(monthly_response) // 6))],
        rotation=45,
        ha='right'
    )
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Average rating with vs without response
    response_impact = df.groupby('has_reply')['score'].mean()

    axes[0, 1].bar(['No Response', 'Has Response'], response_impact.values,
                   color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Average Rating: Response Impact',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Rating')
    axes[0, 1].set_ylim([0, 5])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(response_impact.values):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

    # 3. Response time distribution - CONVERTED TO DAYS
    response_times = df[df['has_reply'] == 1]['response_time_hours'].dropna()
    response_times_days = response_times / 24  # Convert hours to days
    response_times_days = response_times_days[response_times_days < 42]  # Remove outliers (>6 weeks)

    axes[1, 0].hist(response_times_days, bins=50, color='#9b59b6',
                    alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Response Time Distribution',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Response Time (days)')
    axes[1, 0].set_ylabel('Frequency')

    median_days = response_times_days.median()
    mean_days = response_times_days.mean()

    axes[1, 0].axvline(median_days, color='red',
                       linestyle='--', linewidth=2,
                       label=f'Median: {median_days:.1f} days')
    axes[1, 0].axvline(mean_days, color='orange',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {mean_days:.1f} days')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Print summary statistics
    print(f"\n  Response Time Statistics:")
    print(f"    Median: {median_days:.1f} days ({median_days * 24:.1f} hours)")
    print(f"    Mean: {mean_days:.1f} days ({mean_days * 24:.1f} hours)")
    print(f"    Min: {response_times_days.min():.1f} days")
    print(f"    Max: {response_times_days.max():.1f} days")

    # 4. Sentiment by response status
    sentiment_response = pd.crosstab(
        df['has_reply'],
        df['sentiment_label'],
        normalize='index'
    ) * 100

    x = np.arange(2)
    width = 0.25

    axes[1, 1].bar(x - width, sentiment_response['Negative'], width,
                   label='Negative', color=SENTIMENT_COLORS['Negative'])
    axes[1, 1].bar(x, sentiment_response['Neutral'], width,
                   label='Neutral', color=SENTIMENT_COLORS['Neutral'])
    axes[1, 1].bar(x + width, sentiment_response['Positive'], width,
                   label='Positive', color=SENTIMENT_COLORS['Positive'])

    axes[1, 1].set_title('Sentiment Distribution by Response Status',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['No Response', 'Has Response'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('response_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved as 'response_analysis.png'")
    plt.show()


# ============================================================================
# VISUALIZATION 9: COMPREHENSIVE ISSUE BREAKDOWN
# ============================================================================

def create_issue_breakdown(df):
    """
    Detailed breakdown of issues mentioned in reviews using stopword-removed text
    """
    print("\nCreating Issue Breakdown...")

    # Use stopword-removed text for better keyword matching
    text_column = 'content_clean_nostop' if 'content_clean_nostop' in df.columns else 'content_clean'
    negative_reviews = df[df['sentiment_label'] == 'Negative'][text_column]

    issue_keywords = {
        'Login/Authentication': ['login', 'log in', 'sign in', 'signin',
                                 'cant log', "can't log", 'password', 'account'],
        'App Performance': ['crash', 'freeze', 'frozen', 'hang', 'stuck',
                            'loading', 'slow', 'lag'],
        'Billing/Charges': ['charge', 'bill', 'billing', 'payment', 'money', 'fee',
                            'cost', 'expensive', 'price'],
        'Data/Usage': ['data', 'usage', 'balance', 'top up', 'topup',
                       'credit', 'recharge'],
        'Features/UI': ['missing', 'removed', 'gone', 'unavailable',
                        'interface', 'design'],
        'Customer Service': ['support', 'service', 'help', 'customer',
                             'response', 'reply'],
        'Network/Connection': ['network', 'connection', 'signal', 'coverage',
                               'internet', 'wifi']
    }

    issue_counts = {}
    issue_percentages = {}

    for issue, keywords in issue_keywords.items():
        count = negative_reviews.apply(
            lambda x: any(keyword in str(x).lower() for keyword in keywords)
        ).sum()
        issue_counts[issue] = count
        issue_percentages[issue] = (count / len(negative_reviews)) * 100

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Issue frequency bar chart
    issues_sorted = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    issues, counts = zip(*issues_sorted)

    colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.8, len(issues)))

    axes[0].barh(issues, counts, color=colors_gradient, edgecolor='black')
    axes[0].set_title('Issue Frequency in Negative Reviews',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Mentions')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (issue, count) in enumerate(issues_sorted):
        axes[0].text(count + 10, i, f'{count}', va='center', fontweight='bold')

    # 2. Issue percentage breakdown
    percentages_sorted = [issue_percentages[i] for i in issues]

    axes[1].pie(percentages_sorted, labels=issues, autopct='%1.1f%%',
                colors=colors_gradient, startangle=90,
                textprops={'fontsize': 10, 'weight': 'bold'})
    axes[1].set_title('Issue Distribution (% of Negative Reviews)',
                      fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('issue_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved as 'issue_breakdown.png'")
    plt.show()

    return pd.DataFrame({
        'Issue': issues,
        'Count': counts,
        'Percentage': percentages_sorted
    })


# ============================================================================
# MAIN VISUALIZATION PIPELINE
# ============================================================================

def generate_all_visualizations(df, ml_results=None):
    """
    Generate all visualizations for the project
    """
    print("\n" + "=" * 80)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 80)

    # Load ML results if not provided
    if ml_results is None:
        try:
            import pickle
            print("\nLoading ML results from ml_results.pkl...")
            with open('ml_results.pkl', 'rb') as f:
                ml_results = pickle.load(f)
            print("ML results loaded successfully!")
        except FileNotFoundError:
            print("\nWarning: ml_results.pkl not found. Skipping ML visualizations.")
            print("Run the main analysis notebook first to generate ML results.")
            ml_results = None
        except Exception as e:
            print(f"\nError loading ML results: {str(e)}")
            ml_results = None

    # 1. Executive Dashboard
    create_executive_dashboard(df)

    # 2. Sentiment Trends
    create_sentiment_trends(df)

    # 3. Word Clouds
    create_wordclouds(df)

    # 4. Interactive Timeline
    create_interactive_timeline(df)

    # 5. App Version Analysis
    create_version_analysis(df)

    # 6. Response Impact Analysis
    create_response_analysis(df)

    # 7. Issue Breakdown
    issue_df = create_issue_breakdown(df)

    # 8. Model Comparison (if ml_results available)
    if ml_results:
        print("\n" + "=" * 80)
        print("GENERATING ML MODEL VISUALIZATIONS")
        print("=" * 80)

        create_model_comparison(ml_results)

        # Get best model for confusion matrix
        best_model = max(ml_results, key=lambda x: ml_results[x]['f1_score'])
        print(f"\nBest performing model: {best_model}")

        # Extract y_test and predictions from best model
        y_test = ml_results[best_model]['y_test']
        y_pred = ml_results[best_model]['predictions']

        create_confusion_matrix(y_test, y_pred, best_model)

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("1. executive_dashboard.html")
    print("2. sentiment_trends.png")
    print("3. wordclouds.png")
    print("4. interactive_timeline.html")
    print("5. version_analysis.png")
    print("6. response_analysis.png")
    print("7. issue_breakdown.png")
    if ml_results:
        print("8. model_comparison.png")
        print("9. confusion_matrix.png")

    return issue_df


# Example usage:
if __name__ == "__main__":
    # Load your processed data
    print("Loading processed data...")
    df = pd.read_csv('one_nz_processed_reviews.csv')
    print(f"Loaded {len(df)} reviews")

    # Generate all visualizations
    # ML results will be loaded automatically if available
    issue_df = generate_all_visualizations(df)

    print("\n" + "=" * 80)
    print("VISUALIZATION PIPELINE COMPLETE!")
    print("=" * 80)