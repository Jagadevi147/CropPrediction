import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
df=pd.read_csv("C:/Users/Jagadevi/Desktop/ML programs/ICRISAT-District Level Data.csv")
print("Orginal Dataset :")
print(df)
print (df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dtypes)
print(df.describe())

rice=df.groupby('State Name')['RICE PRODUCTION (1000 tons)'].sum()
top_rice=rice.sort_values(ascending=False).head(7)
top_rice=pd.DataFrame(top_rice)
top_rice=top_rice.reset_index()
print(top_rice)

plt.figure(figsize=(12,6))
sns.barplot(x='State Name',y='RICE PRODUCTION (1000 tons)',data=top_rice)
plt.title('Top 7 RICE PRODUCTION State Data',fontsize=18,weight='bold')
plt.xlabel('State Names',fontsize=15,weight='bold')
plt.ylabel('Total_productions',fontsize=15,weight='bold')
plt.xticks(rotation=45,ha='right',fontsize=9,weight='bold')
for i in range(len(top_rice)):
    plt.text(i,top_rice['RICE PRODUCTION (1000 tons)'].iloc[i],
             f"{top_rice['RICE PRODUCTION (1000 tons)'].iloc[i]}",ha='center',fontsize=12,fontweight='bold')
plt.grid(axis='y',linestyle='--')
plt.show()

gom=df.groupby('State Name')['WHEAT PRODUCTION (1000 tons)'].sum()
top_gom=gom.sort_values(ascending=False).head(5)
top_gom=pd.DataFrame(top_gom)
top_gom=top_gom.reset_index()
print(top_gom)


plt.figure(figsize=(8, 8))  # Adjusting the figure size for a square shape
# Pie chart
plt.pie(top_gom['WHEAT PRODUCTION (1000 tons)'], explode=(0.05, 0, 0, 0, 0),
         labels=top_gom['State Name'], colors=['orange', 'gray', 'brown', 'lightyellow', 'hotpink'],
         autopct='%1.1f%%', startangle=70, shadow=True)
plt.title('Wheat Production by Top 5 States %', fontweight='bold', color='navy')
plt.axis('equal')  # Ensures the pie chart is a circle
plt.show()

print(df['State Name'].unique())

oils=df.groupby('State Name')['OILSEEDS PRODUCTION (1000 tons)'].sum()
top_oils=oils.sort_values(ascending=False).head(5)
top_oils=pd.DataFrame(top_oils)
top_oils=top_oils.reset_index()
print(top_oils)

plt.figure(figsize=(10,5))
labels = top_oils['State Name']
sizes = top_oils['OILSEEDS PRODUCTION (1000 tons)']
colors = ['skyblue','yellow','blue','violet','lightgreen'] 
explode = (0.07, 0, 0,0,0) 
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',startangle=90,shadow=True)
plt.title('Oil seed production by top 5 states ',fontweight='bold',color = 'navy')
plt.axis('equal')
plt.tight_layout()
plt.show()


sun=df.groupby('State Name')['SUNFLOWER PRODUCTION (1000 tons)'].sum().sort_values(ascending=False).head(7).reset_index()
print(sun)

plt.figure(figsize=(12,6))
sns.barplot(x='State Name',y='SUNFLOWER PRODUCTION (1000 tons)',data=sun)
plt.title('Top 7 SUNFLOWER PRODUCTION  State ',fontsize=18,weight='bold')
plt.xlabel('State Names',fontsize=15,weight='bold')
plt.ylabel('Total_productions',fontsize=15,weight='bold')
plt.xticks(rotation=45,ha='right',fontsize=9,weight='bold')
for i in range(len(sun)):
    plt.text(i,sun['SUNFLOWER PRODUCTION (1000 tons)'].iloc[i],
             f"{sun['SUNFLOWER PRODUCTION (1000 tons)'].iloc[i]}",ha='center',fontsize=12,fontweight='bold')
plt.grid(axis='y',linestyle='--')
plt.show()

# Prediction for Rice Production with Graph

# Assuming there's a 'Year' column in the dataset to use for predictions
if 'Year' in df.columns:
    # Feature and Target
    X = df[['Year']]  # Feature: Year
    y = df['RICE PRODUCTION (1000 tons)']  # Target: Rice Production

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing and training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))

    # Predicting future production (example: predict for year 2025)
    future_year = np.array([[2025]])
    future_prediction = model.predict(future_year)
    print(f"Predicted Rice Production in 2025: {future_prediction[0]:.2f} thousand tons")

    # Plotting the actual vs predicted production
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.scatter(X, y, color='blue', label='Actual Data')
    
    # Plot predictions for the test data
    plt.scatter(X_test, y_pred, color='red', label='Predicted Data')
    
    # Line plot for overall prediction
    X_range = np.arange(X['Year'].min(), 2026).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    plt.plot(X_range, y_range_pred, color='green', label='Prediction Line')

    # Plotting the predicted point for 2025
    plt.scatter(2025, future_prediction, color='purple', label='Prediction for 2025', marker='X', s=100)

    plt.title('Rice Production Prediction (1000 tons)', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    plt.ylabel('Rice Production (1000 tons)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("The dataset does not have a 'Year' column for predictions.")
