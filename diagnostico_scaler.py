import joblib
import numpy as np

# Cargar el scaler
scaler = joblib.load('min_max_scaler.joblib')

print("=== DIAGNÓSTICO DEL SCALER ===\n")
print(f"Número de features en el scaler: {scaler.n_features_in_}")
print(f"\nMín valores por feature:")
print(scaler.data_min_)
print(f"\nMáx valores por feature:")
print(scaler.data_max_)
print(f"\nRango del feature 2 (quantity_available):")
print(f"  Min: {scaler.data_min_[2]}")
print(f"  Max: {scaler.data_max_[2]}")
print(f"  Rango: {scaler.data_max_[2] - scaler.data_min_[2]}")

# Prueba de desescalado
print("\n=== PRUEBA DE DESESCALADO ===")
test_value_scaled = 0.05  # Un valor escalado de prueba
dummy = np.zeros((1, 18))
dummy[0, 2] = test_value_scaled
test_value_real = scaler.inverse_transform(dummy)[0, 2]
print(f"Valor escalado: {test_value_scaled}")
print(f"Valor real: {test_value_real}")
