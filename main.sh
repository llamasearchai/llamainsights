#!/bin/bash
#
# Agile Insights Installation and Execution Script
#
# This script installs, configures, and executes the Agile Insights
# application, a tool for marketing funnel analysis and optimization.
#
# Author: [Your Name]
# Date: 2024-01-24

# --- Configuration ---
APP_NAME="agileinsights"
VERSION="1.0.0"
PYTHON_VERSION="3.9"
VENV_DIR=".venv"
REPORT_DIR="agileinsights_reports"
DATA_DIR="$HOME/$APP_NAME-data"
OUTPUT_DIR="$HOME/$APP_NAME-output"
LOG_FILE="${APP_NAME}.log"

# --- Helper Functions ---

function log_info {
  echo -e "\e[34m[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1\e[0m" | tee -a "$LOG_FILE"
}

function log_success {
  echo -e "\e[32m[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1\e[0m" | tee -a "$LOG_FILE"
}

function log_warning {
  echo -e "\e[33m[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1\e[0m" | tee -a "$LOG_FILE"
}

function log_error {
  echo -e "\e[31m[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1\e[0m" | tee -a "$LOG_FILE"
}

function check_command {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed. Please install it."
        exit 1
    fi
}

function create_venv {
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment in $VENV_DIR"
        python3 -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment."
            exit 1
        }
    else
        log_info "Virtual environment already exists in $VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate" || {
        log_error "Failed to activate virtual environment."
        exit 1
    }
}

function install_dependencies {
    log_info "Installing dependencies using pip"
    pip install --upgrade pip setuptools wheel || {
        log_error "Failed to upgrade pip."
        exit 1
    }
    pip install pandas numpy matplotlib seaborn faker colorama tqdm scipy tabulate || {
        log_error "Failed to install dependencies."
        exit 1
    }
    log_success "Dependencies installed successfully."
}

function cleanup {
    log_info "Cleaning up: removing virtual environment"
    deactivate &> /dev/null || log_warning "Failed to deactivate venv during cleanup."
    rm -rf "$VENV_DIR"
}

function install_application_files {
    log_info "Installing Agile Insights application files"

    # Create a directory for app files
    mkdir -p "$DATA_DIR" || {
        log_error "Failed to create data directory."
        exit 1
    }
	mkdir -p "$REPORT_DIR" || {
        log_error "Failed to create report directory."
		exit 1
	}

    cat <<'EOF' > agileinsights.py
#!/usr/bin/env python3
"""
Agile Insights: Advanced Marketing Funnel Analysis CLI
"""

import argparse
import os
import sys
import json
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from colorama import init, Fore, Style
from tqdm import tqdm
from scipy import stats
import webbrowser
from pathlib import Path
import warnings

# Initialize colorama
init(autoreset=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set(style="darkgrid")

LOGO = f"""
{Fore.GREEN}██████╗ ███████╗███╗   ██╗██████╗  ██████╗ ███████╗██████╗ {Fore.RESET}
{Fore.GREEN}██╔═══██╗██╔════╝████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗{Fore.RESET}
{Fore.GREEN}██║   ██║███████╗██╔██╗ ██║██║  ██║██║  ███╗███████╗██████╔╝{Fore.RESET}
{Fore.CYAN}██║   ██║╚════██║██║╚██╗██║██║  ██║██║   ██║╚════██║██╔══██╗{Fore.RESET}
{Fore.CYAN}╚██████╔╝███████║██║ ╚████║██████╔╝╚██████╔╝███████║██║  ██║{Fore.RESET}
{Fore.CYAN} ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝{Fore.RESET}
{Fore.YELLOW}Agile Insights v1.0.0{Fore.RESET} - {Fore.BLUE}Unlock marketing potential{Fore.RESET}
"""

class AgileInsights:
    def __init__(self):
        self.data = None
        self.funnel_stages = None
        self.report_dir = "agileinsights_reports"
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
    
    def load_data(self, file_path):
        """Load data from CSV, Excel, or JSON file"""
        print(f"{Fore.YELLOW}Loading data from {file_path}...{Fore.RESET}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                self.data = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            else:
                print(f"{Fore.RED}Unsupported file format: {file_extension}{Fore.RESET}")
                sys.exit(1)
                
            print(f"{Fore.GREEN}Successfully loaded {len(self.data)} records{Fore.RESET}")
            print(f"\n{Fore.CYAN}Data Preview:{Fore.RESET}")
            print(tabulate(self.data.head(), headers='keys', tablefmt='pretty'))
            
            return True
        except Exception as e:
            print(f"{Fore.RED}Error loading data: {str(e)}{Fore.RESET}")
            return False
    
    def generate_sample_data(self, num_records=1000):
        """Generate sample funnel data for demonstration"""
        print(f"{Fore.YELLOW}Generating sample funnel data with {num_records} records...{Fore.RESET}")
        
        # Define traffic sources
        traffic_sources = ['Organic Search', 'Paid Search', 'Social Media', 'Direct', 'Referral', 'Email']
        source_weights = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
        
        # Define landing pages
        landing_pages = ['Home', 'Product', 'Pricing', 'Blog', 'Documentation', 'Signup', 'Demo']
        page_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]
        
        # Conversion probabilities by stage
        conversion_rates = {
            'Visitor_to_PageView': 0.85,
            'PageView_to_Signup': 0.15,
            'Signup_to_MQL': 0.40,
            'MQL_to_Demo': 0.30,
            'Demo_to_Opportunity': 0.60,
            'Opportunity_to_Customer': 0.25
        }
        
        # Generate dates covering last 30 days
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
        date_range = [start_date + datetime.timedelta(days=x) for x in range(31)]
        
        # Initialize lists to hold the data
        sessions = []
        current_id = 10000
        
        # Generate session data
        for _ in tqdm(range(num_records), desc="Generating Sessions"):
            current_id += 1
            session_id = f"session_{current_id}"
            date = random.choice(date_range)
            traffic_source = np.random.choice(traffic_sources, p=source_weights)
            landing_page = np.random.choice(landing_pages, p=page_weights)
            
            # Basic session data
            session_data = {
                'session_id': session_id,
                'date': date,
                'traffic_source': traffic_source,
                'landing_page': landing_page,
                'visitor': True,
                'pageview': np.random.random() < conversion_rates['Visitor_to_PageView'],
            }
            
            # Continue the funnel based on conditional probabilities
            session_data['signup'] = session_data['pageview'] and (np.random.random() < conversion_rates['PageView_to_Signup'])
            session_data['mql'] = session_data['signup'] and (np.random.random() < conversion_rates['Signup_to_MQL'])
            session_data['demo'] = session_data['mql'] and (np.random.random() < conversion_rates['MQL_to_Demo'])
            session_data['opportunity'] = session_data['demo'] and (np.random.random() < conversion_rates['Demo_to_Opportunity'])
            session_data['customer'] = session_data['opportunity'] and (np.random.random() < conversion_rates['Opportunity_to_Customer'])
            
            # Add time spent and additional metrics
            session_data['time_spent'] = np.random.normal(3 if session_data['pageview'] else 0.5, 1)
            session_data['time_spent'] = max(0.1, session_data['time_spent'])  # Ensure positive time
            
            # Add company size and industry for segmentation
            company_sizes = ['Small', 'Medium', 'Enterprise']
            industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
            session_data['company_size'] = np.random.choice(company_sizes)
            session_data['industry'] = np.random.choice(industries)
            
            sessions.append(session_data)
        
        # Convert to DataFrame
        self.data = pd.DataFrame(sessions)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Set funnel stages
        self.funnel_stages = ['visitor', 'pageview', 'signup', 'mql', 'demo', 'opportunity', 'customer']
        
        print(f"{Fore.GREEN}Successfully generated sample data with {len(self.data)} records{Fore.RESET}")
        print(f"\n{Fore.CYAN}Data Preview:{Fore.RESET}")
        print(tabulate(self.data.head(), headers='keys', tablefmt='pretty'))
        
        return True

    def analyze_funnel(self):
        """Analyze the conversion funnel and visualize drop-offs"""
        if self.data is None:
            print(f"{Fore.RED}No data loaded. Please load data first.{Fore.RESET}")
            return False
        
        print(f"{Fore.YELLOW}Analyzing conversion funnel...{Fore.RESET}")
        
        # Identify funnel stages if not set
        if self.funnel_stages is None:
            # Try to automatically detect funnel stages from boolean columns
            bool_columns = self.data.select_dtypes(include=['bool']).columns.tolist()
            if bool_columns:
                self.funnel_stages = bool_columns
                print(f"{Fore.GREEN}Automatically detected funnel stages: {', '.join(self.funnel_stages)}{Fore.RESET}")
            else:
                print(f"{Fore.RED}Could not automatically detect funnel stages. Please specify them.{Fore.RESET}")
                return False
        
        # Count users at each stage of the funnel
        funnel_counts = []
        for stage in self.funnel_stages:
            count = self.data[stage].sum()
            funnel_counts.append(count)
            
        # Calculate conversion rates between stages
        conversion_rates = []
        for i in range(len(funnel_counts) - 1):
            if funnel_counts[i] > 0:
                rate = (funnel_counts[i+1] / funnel_counts[i]) * 100
            else:
                rate = 0
            conversion_rates.append(rate)
        
        # Create and save funnel visualization
        plt.figure(figsize=(12, 8))
        
        # Funnel chart
        plt.subplot(2, 1, 1)
        sns.barplot(x=self.funnel_stages, y=funnel_counts, palette='viridis')
        plt.title('Conversion Funnel', fontsize=16)
        plt.xlabel('Funnel Stage', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, count in enumerate(funnel_counts):
            plt.text(i, count + (max(funnel_counts) * 0.02), 
                    f"{int(count)}", 
                    ha='center', va='bottom', fontsize=10)
        
        # Conversion rates
        plt.subplot(2, 1, 2)
        stages_pairs = [f"{self.funnel_stages[i]} → {self.funnel_stages[i+1]}" 
                       for i in range(len(self.funnel_stages)-1)]
        
        sns.barplot(x=stages_pairs, y=conversion_rates, palette='plasma')
        plt.title('Conversion Rates Between Stages', fontsize=16)
        plt.xlabel('Stage Transition', fontsize=12)
        plt.ylabel('Conversion Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, rate in enumerate(conversion_rates):
            plt.text(i, rate + 1, f"{rate:.1f}%", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        report_path = os.path.join(self.report_dir, "funnel_analysis.png")
        plt.savefig(report_path)
        
        print(f"{Fore.GREEN}Funnel analysis complete! Report saved to {report_path}{Fore.RESET}")
        
        # Display funnel statistics
        print(f"\n{Fore.CYAN}Funnel Statistics:{Fore.RESET}")
        stage_df = pd.DataFrame({
            'Stage': self.funnel_stages,
            'Count': funnel_counts,
            'Conversion Rate (%)': ['N/A'] + [f"{rate:.1f}%" for rate in conversion_rates]
        })
        print(tabulate(stage_df, headers='keys', tablefmt='pretty'))
        
        # Calculate overall funnel conversion (first to last)
        if funnel_counts[0] > 0:
            overall_conversion = (funnel_counts[-1] / funnel_counts[0]) * 100
            print(f"\n{Fore.YELLOW}Overall Funnel Conversion ({self.funnel_stages[0]} to {self.funnel_stages[-1]}): "
                  f"{overall_conversion:.2f}%{Fore.RESET}")
        
        return True

    def segment_analysis(self, segment_column):
        """Analyze the funnel performance by segment"""
        if self.data is None or self.funnel_stages is None:
            print(f"{Fore.RED}No data loaded. Please load data first.{Fore.RESET}")
            return False
        
        if segment_column not in self.data.columns:
            print(f"{Fore.RED}Segment column '{segment_column}' not found in data.{Fore.RESET}")
            return False
        
        print(f"{Fore.YELLOW}Analyzing funnel by {segment_column}...{Fore.RESET}")
        
        # Get unique segments
        segments = self.data[segment_column].unique()
        
        # Create a figure with subplots
        fig, axes = plt.subplots(len(segments), 1, figsize=(12, 4 * len(segments)))
        
        # If there's only one segment, axes is not a list
        if len(segments) == 1:
            axes = [axes]
        
        # Overall conversion rates
        segment_overall_rates = {}
        
        # Analyze each segment
        for i, segment in enumerate(segments):
            segment_data = self.data[self.data[segment_column] == segment]
            
            # Count users at each stage of the funnel for this segment
            funnel_counts = []
            for stage in self.funnel_stages:
                count = segment_data[stage].sum()
                funnel_counts.append(count)
            
            # Calculate conversion rates between stages
            conversion_rates = []
            for j in range(len(funnel_counts) - 1):
                if funnel_counts[j] > 0:
                    rate = (funnel_counts[j+1] / funnel_counts[j]) * 100
                else:
                    rate = 0
                conversion_rates.append(rate)
            
            # Calculate overall conversion rate
            if funnel_counts[0] > 0:
                overall_rate = (funnel_counts[-1] / funnel_counts[0]) * 100
                segment_overall_rates[segment] = overall_rate
            else:
                segment_overall_rates[segment] = 0
            
            # Plot segment funnel
            sns.barplot(x=self.funnel_stages, y=funnel_counts, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Conversion Funnel: {segment_column} = {segment}', fontsize=14)
            axes[i].set_xlabel('Funnel Stage', fontsize=12)
            axes[i].set_ylabel('Number of Users', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for j, count in enumerate(funnel_counts):
                axes[i].text(j, count + (max(funnel_counts) * 0.02) if max(funnel_counts) > 0 else 1, 
                        f"{int(count)}", 
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the figure
        report_path = os.path.join(self.report_dir, f"segment_analysis_{segment_column}.png")
        plt.savefig(report_path)
        
        print(f"{Fore.GREEN}Segment analysis complete! Report saved to {report_path}{Fore.RESET}")
        
        # Create a comparison chart of overall conversion rates by segment
        plt.figure(figsize=(10, 6))
        segments_list = list(segment_overall_rates.keys())
        rates_list = list(segment_overall_rates.values())
        
        # Sort segments by conversion rate
        sorted_indices = np.argsort(rates_list)[::-1]  # Descending order
        sorted_segments = [segments_list[i] for i in sorted_indices]
        sorted_rates = [rates_list[i] for i in sorted_indices]
        
        sns.barplot(x=sorted_segments, y=sorted_rates, palette='viridis')
        plt.title(f'Overall Conversion Rate by {segment_column}', fontsize=16)
        plt.xlabel(segment_column, fontsize=12)
        plt.ylabel('Overall Conversion Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, rate in enumerate(sorted_rates):
            plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the comparison figure
        comparison_path = os.path.join(self.report_dir, f"segment_comparison_{segment_column}.png")
        plt.savefig(comparison_path)
        
        print(f"{Fore.GREEN}Segment comparison chart saved to {comparison_path}{Fore.RESET}")
        
        # Display segment comparison table
        print(f"\n{Fore.CYAN}Segment Comparison ({segment_column}):{Fore.RESET}")
        segment_df = pd.DataFrame({
            segment_column: sorted_segments,
            'Overall Conversion Rate (%)': [f"{rate:.1f}%" for rate in sorted_rates]
        })
        print(tabulate(segment_df, headers='keys', tablefmt='pretty'))
        
        return True

    def ab_test_simulation(self, control_size=5000, treatment_size=5000, improvement=15):
        """Simulate an A/B test for conversion rate optimization"""
        if self.data is None or self.funnel_stages is None:
            print(f"{Fore.RED}No data or funnel stages defined. Please load data and analyze funnel first.{Fore.RESET}")
            return False
        
        print(f"{Fore.YELLOW}Simulating A/B test with {control_size} control users and {treatment_size} treatment users...{Fore.RESET}")
        print(f"{Fore.YELLOW}Expected improvement: {improvement}%{Fore.RESET}")
        
        # Get baseline conversion rate from the first to the second stage
        baseline_conversion = (self.data[self.funnel_stages[1]].sum() / self.data[self.funnel_stages[0]].sum()) * 100
        
        # Set improved conversion rate for treatment
        treatment_conversion = baseline_conversion * (1 + (improvement / 100))
        
        # Ensure treatment_conversion is not above 100%
        treatment_conversion = min(treatment_conversion, 99.9)
        
        # Simulate control and treatment groups
        control_conversions = np.random.binomial(1, baseline_conversion / 100, control_size)
        treatment_conversions = np.random.binomial(1, treatment_conversion / 100, treatment_size)
        
        # Calculate results
        control_rate = (control_conversions.sum() / control_size) * 100
        treatment_rate = (treatment_conversions.sum() / treatment_size) * 100
        
        # Statistical test
        z_score, p_value = stats.proportions_ztest(
            [treatment_conversions.sum(), control_conversions.sum()], 
            [treatment_size, control_size]
        )
        
        # Determine if result is significant
        alpha = 0.05
        is_significant = p_value < alpha
        
        # Calculate relative improvement
        relative_improvement = ((treatment_rate - control_rate) / control_rate) * 100
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Conversion rates comparison
        plt.subplot(2, 1, 1)
        bars = plt.bar(['Control', 'Treatment'], [control_rate, treatment_rate], color=['blue', 'green'])
        plt.title('A/B Test Results: Conversion Rate Comparison', fontsize=16)
        plt.ylabel('Conversion Rate (%)', fontsize=12)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.2f}%", ha='center', va='bottom', fontsize=12)
        
        # Mark if significant
        significance_text = f"p-value: {p_value:.4f}" + (" (Significant)" if is_significant else " (Not Significant)")
        plt.text(0.5, max(control_rate, treatment_rate) * 1.2, significance_text, 
                ha='center', va='center', fontsize=14, 
                color='green' if is_significant else 'red')
        
        # User counts
        plt.subplot(2, 1, 2)
        control_users = np.array([control_size, control_conversions.sum()])
        treatment_users = np.array([treatment_size, treatment_conversions.sum()])
        
        index = np.arange(2)
        bar_width = 0.35
        
        plt.bar(index, control_users, bar_width, label='Control', color='blue', alpha=0.7)
        plt.bar(index + bar_width, treatment_users, bar_width, label='Treatment', color='green', alpha=0.7)
        
        plt.xlabel('Stage', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.title('User Counts by Stage', fontsize=16)
        plt.xticks(index + bar_width/2, ['Total Users', 'Converted Users'])
        plt.legend()
        
        # Add counts
        for i, count in enumerate(control_users):
            plt.text(i, count + 50, str(int(count)), ha='center', va='bottom', color='blue', fontsize=12)
        
        for i, count in enumerate(treatment_users):
            plt.text(i + bar_width, count + 50, str(int(count)), ha='center', va='bottom', color='green', fontsize=12)
        
        plt.tight_layout()
        
        # Save the figure
        report_path = os.path.join(self.report_dir, "ab_test_simulation.png")
        plt.savefig(report_path)
        
        print(f"{Fore.GREEN}A/B test simulation complete! Report saved to {report_path}{Fore.RESET}")
        
        # Display test results
        print(f"\n{Fore.CYAN}A/B Test Results:{Fore.RESET}")
        test_df = pd.DataFrame({
            'Metric': ['Control Size', 'Treatment Size', 'Control Conversion Rate', 'Treatment Conversion Rate', 
                      'Absolute Difference', 'Relative Improvement', 'p-value', 'Statistically Significant'],
            'Value': [control_size, treatment_size, f"{control_rate:.2f}%", f"{treatment_rate:.2f}%", 
                     f"{treatment_rate - control_rate:.2f} pp", f"{relative_improvement:.2f}%", 
                     f"{p_value:.4f}", "Yes" if is_significant else "No"]
        })
        print(tabulate(test_df, headers='keys', tablefmt='pretty'))
        
        # Calculate potential impact
        if is_significant and treatment_rate > control_rate:
            total_visitors = self.data[self.funnel_stages[0]].sum()
            current_conversions = self.data[self.funnel_stages[1]].sum()
            potential_conversions = total_visitors * (treatment_rate / 100)
            additional_conversions = potential_conversions - current_conversions
            
            print(f"\n{Fore.GREEN}Potential Impact If Implemented:{Fore.RESET}")
            print(f"• Current monthly conversions: {int(current_conversions)}")
            print(f"• Potential monthly conversions: {int(potential_conversions)}")
            print(f"• Additional monthly conversions: {int(additional_conversions)} (+{relative_improvement:.1f}%)")
        
        return True

    def generate_report(self):
        """Generate a comprehensive HTML report of all analyses"""
        if self.data is None:
            print(f"{Fore.RED}No data loaded. Please load data first.{Fore.RESET}")
            return False
        
        print(f"{Fore.YELLOW}Generating comprehensive HTML report...{Fore.RESET}")
        
        # Create report template
        html_template = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Agile Insights Marketing Funnel Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3, h4 {
                    color: #2c3e50;
                }
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                    color: white;
                    border-radius:.4rem;
                }
                .section {
                    margin-bottom: 40px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: .4rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .chart {
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    border-radius: 4px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .metric-card {
                    background-color: white;
                    border-radius: 4px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .metric-value {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .metric-label {
                    font-size: 0.9rem;
                    color: #7f8c8d;
                }
                .flex-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: space-between;
                }
                .flex-item {
                    flex: 1 1 calc(33% - 20px);
                    min-width: 250px;
                }
                .recommendations {
                    background-color: #e3f2fd;
                    padding: 15px;
                    border-left: 4px solid #2196f3;
                    margin: 20px 0;
                }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #7f8c8d;
                    font-size: 0.9rem;
                }
            </style>
        </head>
        <body>
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report provides a comprehensive analysis of your marketing funnel performance. Key insights are highlighted below.</p>
                
                <div class="flex-container">
                    <div class="flex-item metric-card">
                        <div class="metric-label">Overall Conversion Rate</div>
                        <div class="metric-value">{overall_conversion}%</div>
                    </div>
                    <div class="flex-item metric-card">
                        <div class="metric-label">Total Visitors</div>
                        <div class="metric-value">{total_visitors}</div>
                    </div>
                    <div class="flex-item metric-card">
                        <div class="metric-label">Customers Acquired</div>
                        <div class="metric-value">{total_customers}</div>
                    </div>
                </div>
            </div>
            
            <div class="header">
                <h1>Agile Insights Marketing Funnel Analysis Report</h1>
                <p>Generated on {date}</p>
            </div>
            
            <div class="section">
                <h2>Funnel Analysis</h2>
                <p>Below is a breakdown of your marketing funnel performance, showing user volume and conversion rates at each stage.</p>
                
                <img src="funnel_analysis.png" alt="Funnel Analysis" class="chart">
                
                <h3>Funnel Statistics</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Stage</th>
                            <th>Count</th>
                            <th>Conversion Rate (%)</th>
                            <th>Drop-off (%)</th>