import pandas as pd

from datase import prepare_dataset


def test_prepare_dataset_creates_expected_columns_and_valid_values():
    df = pd.DataFrame([
        {
            'OrderID': 'A-1',
            'CustomerID': 'C-1',
            'Gender': 'Male',
            'Age': 28,
            'City': 'New York',
            'ProductCategory': 'Electronics',
            'ProductName': 'Laptop',
            'Quantity': 2,
            'Price': 800,
            'OrderDate': '2024-01-15',
            'PaymentMethod': 'Credit Card',
            'ShippingCost': 25,
            'Rating': 4.5,
            'DiscountPercent': 10,
        },
        {
            'OrderID': 'A-2',
            'CustomerID': 'C-2',
            'Gender': 'Female',
            'Age': 35,
            'City': 'Chicago',
            'ProductCategory': 'Home',
            'ProductName': 'Lamp',
            'Quantity': 3,
            'Price': -200,
            'OrderDate': '2024-02-20',
            'PaymentMethod': 'Debit Card',
            'ShippingCost': 15,
            'Rating': 3,
            'DiscountPercent': 20,
        },
    ])

    result = prepare_dataset(df)

    assert {'TotalAmount', 'OrderMonth', 'OrderYear', 'AgeGroup', 'Price_Capped', 'Quantity_Capped'} <= set(result.columns)
    assert (result['Quantity'] > 0).all()
    assert (result['Rating'].between(1, 5)).all()
    assert result['TotalAmount'].ge(0).all()
