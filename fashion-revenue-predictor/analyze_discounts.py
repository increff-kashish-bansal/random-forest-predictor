import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discount_analysis.log'),
        logging.StreamHandler()
    ]
)

def analyze_discounts(file_path):
    """Analyze discount calculations in sales data."""
    try:
        # Read the sales data
        df_sales = pd.read_csv(file_path)
        logging.info(f"Loaded sales data with shape: {df_sales.shape}")
        logging.info(f"Columns in the data: {df_sales.columns.tolist()}")
        
        # Calculate discount percentage
        df_sales['disc_perc'] = np.where(
            (df_sales['revenue'] + df_sales['disc_value']) > 0,
            df_sales['disc_value'] * 100 / (df_sales['revenue'] + df_sales['disc_value']),
            0
        )
        df_sales['disc_perc'] = df_sales['disc_perc'].clip(0, 100)
        
        # Basic statistics
        logging.info("\nBasic Statistics:")
        logging.info(f"Total number of sales: {len(df_sales)}")
        logging.info(f"Number of discounted sales: {len(df_sales[df_sales['disc_perc'] > 0])}")
        logging.info(f"Average revenue: ₹{df_sales['revenue'].mean():,.2f}")
        logging.info(f"Average discount value: ₹{df_sales['disc_value'].mean():,.2f}")
        logging.info(f"Median discount value: ₹{df_sales['disc_value'].median():,.2f}")
        
        # Discount percentage statistics
        logging.info("\nDiscount Percentage Statistics:")
        logging.info(f"Min discount: {df_sales['disc_perc'].min():.2%}")
        logging.info(f"Max discount: {df_sales['disc_perc'].max():.2%}")
        logging.info(f"Mean discount: {df_sales['disc_perc'].mean():.2%}")
        logging.info(f"Median discount: {df_sales['disc_perc'].median():.2%}")
        
        # Distribution of discount values
        logging.info("\nDiscount Value Distribution:")
        logging.info(df_sales['disc_value'].describe())
        
        # Distribution of discount percentages
        logging.info("\nDiscount Percentage Distribution:")
        logging.info(df_sales['disc_perc'].describe())
        
        # Analyze high discount cases
        high_discounts = df_sales[df_sales['disc_perc'] > 50]
        logging.info(f"\nNumber of sales with >50% discount: {len(high_discounts)}")
        if not high_discounts.empty:
            logging.info("\nSample of High Discount Sales (>50%):")
            # Get all available columns for display
            display_cols = [col for col in ['store', 'revenue', 'disc_value', 'disc_perc'] if col in df_sales.columns]
            logging.info(high_discounts[display_cols].head())
            
            # Analyze revenue vs discount value for high discount cases
            logging.info("\nRevenue vs Discount Value for High Discount Cases:")
            logging.info(f"Average revenue for high discount cases: ₹{high_discounts['revenue'].mean():,.2f}")
            logging.info(f"Average discount value for high discount cases: ₹{high_discounts['disc_value'].mean():,.2f}")
            logging.info(f"Median revenue for high discount cases: ₹{high_discounts['revenue'].median():,.2f}")
            logging.info(f"Median discount value for high discount cases: ₹{high_discounts['disc_value'].median():,.2f}")
            
            # Additional analysis for high discount cases
            logging.info("\nHigh Discount Cases Analysis:")
            logging.info(f"Number of cases where discount > revenue: {len(high_discounts[high_discounts['disc_value'] > high_discounts['revenue']])}")
            logging.info(f"Number of cases with zero revenue: {len(high_discounts[high_discounts['revenue'] == 0])}")
        
        # Check for potential data quality issues
        logging.info("\nData Quality Checks:")
        logging.info(f"Number of zero revenue sales: {len(df_sales[df_sales['revenue'] == 0])}")
        logging.info(f"Number of negative revenue sales: {len(df_sales[df_sales['revenue'] < 0])}")
        logging.info(f"Number of negative discount values: {len(df_sales[df_sales['disc_value'] < 0])}")
        
        # Analyze discount ranges
        logging.info("\nDiscount Range Analysis:")
        ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
        for start, end in ranges:
            count = len(df_sales[(df_sales['disc_perc'] > start) & (df_sales['disc_perc'] <= end)])
            logging.info(f"Discounts between {start}% and {end}%: {count} sales ({count/len(df_sales)*100:.1f}%)")
        
        # Save detailed analysis to CSV
        analysis_file = 'discount_analysis.csv'
        df_sales[display_cols].to_csv(analysis_file, index=False)
        logging.info(f"\nDetailed analysis saved to {analysis_file}")
        
    except Exception as e:
        logging.error(f"Error analyzing discounts: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_discounts.py <sales_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_discounts(file_path) 