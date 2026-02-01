import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

file_path = "data/Private Domestic - Average Rents by Class (from 1982).xlsx"


try:
    df = pd.read_excel(file_path, sheet_name=0, header=4)
    month_col = df.columns[0]
    
    numeric_cols = []
    for col in df.columns:
        if col != month_col:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    avg_val = df[col].mean()
                    if 50 < avg_val < 1000:  
                        numeric_cols.append(col)
            except:
                pass
    n_months = len(df)
    start_date = datetime(1982, 1, 1)
    
    dates = pd.date_range(start=start_date, periods=n_months, freq='ME')
    df['Date'] = dates
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
        
    new_names = []
    class_letters = ['A', 'B', 'C', 'D', 'E']
    districts = ['Hong_Kong_Island', 'Kowloon', 'New_Territories']
    
    for i, col in enumerate(numeric_cols):
        class_idx = (i // 3) % 5 
        district_idx = i % 3     
        new_name = f"Class_{class_letters[class_idx]}_{districts[district_idx]}"
        new_names.append(new_name)
    col_mapping = {old: new for old, new in zip(numeric_cols, new_names)}
    df = df.rename(columns=col_mapping)
        
    for col in new_names:
        df[col] = df[col].ffill().bfill()
        
    for class_letter in class_letters[:3]:  
        class_cols = [col for col in new_names if f'Class_{class_letter}' in col]
        if class_cols:
            for col in class_cols:
                avg = df[col].mean()
                growth = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100
    
    plt.figure(figsize=(14, 8))
    for i, class_letter in enumerate(class_letters[:3]):
        class_cols = [col for col in new_names if f'Class_{class_letter}' in col]
        if class_cols:
            class_avg = df[class_cols].mean(axis=1)
            plt.plot(df['Date'], class_avg, label=f'Class {class_letter}', linewidth=2)
    
    plt.title('Hong Kong Average Rental Prices by Property Class', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Monthly Rent per m² (HKD)', fontsize=12)
    plt.legend(title='Property Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hk_rental_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    output_file = 'hk_rental_cleaned.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
except Exception as e:
    print(f"\nError: {e}")
    try:
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
        
    except Exception as e2:
        print(f"Also failed: {e2}")
