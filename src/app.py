#from utils import db_connect
#engine = db_connect()

# your code here

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import os # Para os.path y verificar si el archivo existe

# --- Configuración de la Aplicación Flask ---
# Obtener la ruta absoluta del directorio donde se encuentra app.py (asumimos que es src/)
# y decirle a Flask que busque templates aquí.
src_dir_path = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=src_dir_path)

# --- Cargar los Artefactos del Modelo desde el Paquete Único ---
MODEL_PACKAGE_FILE = 'price_prediction_package_v1.pkl' # Asumiendo que está en src/
full_model_package_path = os.path.join(src_dir_path, MODEL_PACKAGE_FILE)

# Inicializar variables globales para los artefactos
model = None
scaler = None
city_encoder = None
model_columns = []
city_options = []
PREDICTION_YEAR = 2014 # Valor por defecto, se intentará cargar desde el paquete
TARGET_COLUMN_NAME = 'price_log' # Valor por defecto

try:
    if os.path.exists(full_model_package_path):
        with open(full_model_package_path, 'rb') as file:
            loaded_package = pickle.load(file)

        # Extraer cada artefacto del diccionario cargado
        model = loaded_package.get('model')
        scaler = loaded_package.get('scaler')
        city_encoder = loaded_package.get('city_encoder')
        model_columns = loaded_package.get('model_columns_ordered', []) # Clave usada al guardar
        city_options = loaded_package.get('city_options_for_dropdown', []) # Clave usada al guardar
        
        if 'feature_engineering_details' in loaded_package and \
           'prediction_year_reference' in loaded_package['feature_engineering_details']:
            PREDICTION_YEAR = loaded_package['feature_engineering_details']['prediction_year_reference']
        
        if 'target_variable' in loaded_package:
            TARGET_COLUMN_NAME = loaded_package['target_variable']

        # Verificación crucial de que los componentes esenciales se cargaron
        if not all([model, scaler, city_encoder, model_columns]):
            # city_options podría estar legítimamente vacía si no hay ciudades, pero los otros son críticos
            raise ValueError("Error: Uno o más artefactos esenciales (modelo, scaler, encoder, model_columns) no se encontraron o no se cargaron correctamente del paquete .pkl.")
        
        print(f"Paquete de modelo '{MODEL_PACKAGE_FILE}' cargado exitosamente desde '{src_dir_path}'.")
        print(f"  - Usando PREDICTION_YEAR: {PREDICTION_YEAR}")
        print(f"  - Usando TARGET_COLUMN_NAME: {TARGET_COLUMN_NAME}")
        print(f"  - Columnas del modelo cargadas: {len(model_columns)} (Primeras 5: {model_columns[:5]}...)")
        print(f"  - Opciones de ciudad cargadas: {len(city_options)}")

    else:
        # Este es un error crítico si el archivo no existe
        error_message = f"Error Crítico: El archivo del paquete del modelo '{full_model_package_path}' no fue encontrado. La aplicación no puede funcionar sin él."
        print(error_message)
        # En un escenario real, podrías querer que la app no inicie o muestre un error persistente.
        # Por ahora, las variables de artefactos permanecerán como None o vacías.
        
except Exception as e:
    print(f"Error Crítico al cargar o procesar el paquete del modelo desde '{full_model_package_path}': {e}")
    import traceback
    traceback.print_exc()
    # Similar al FileNotFoundError, la app podría no funcionar.

# --- Rutas de la Aplicación ---
@app.route('/', methods=['GET'])
def home():
    # Verificar si los artefactos esenciales se cargaron antes de intentar renderizar la página
    if not model or not city_encoder or not scaler or not model_columns:
        return "Error: El modelo o sus componentes no se cargaron correctamente. Por favor, revise los logs del servidor."
    
    # Asegurarse de que city_options sea una lista, incluso si está vacía, para el template
    current_city_options = city_options if city_options else []
    
    return render_template('index.html', 
                           city_options=sorted(current_city_options), 
                           PREDICTION_YEAR=PREDICTION_YEAR,
                           form_data={}) # Pasar form_data vacío para el primer renderizado

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not city_encoder or not model_columns:
        return render_template('index.html', 
                               prediction_text='Error: El modelo o sus componentes no están cargados. Revisa la consola.', 
                               city_options=sorted(city_options if city_options else []), 
                               PREDICTION_YEAR=PREDICTION_YEAR)

    try:
        form_data = request.form.to_dict()
        
        # Recopilar y convertir datos del formulario
        features_input = {
            'bedrooms': float(form_data['bedrooms']),
            'bathrooms': float(form_data['bathrooms']),
            'sqft_living': int(form_data['sqft_living']),
            'sqft_lot': int(form_data['sqft_lot']),
            'floors': float(form_data['floors']),
            'waterfront': int(form_data['waterfront']),
            'view': int(form_data['view']),
            'condition': int(form_data['condition']),
            'sqft_above': int(form_data['sqft_above']),
            'sqft_basement': int(form_data['sqft_basement']),
            'yr_built': int(form_data['yr_built']),
            'yr_renovated': int(form_data.get('yr_renovated', 0)), # Usar .get con default para yr_renovated
            'city': form_data['city'],
            'sale_month': int(form_data['sale_month']),
        }

        # Ingeniería de características (consistente con el entrenamiento)
        features_input['age_at_sale'] = PREDICTION_YEAR - features_input['yr_built']
        if features_input['yr_renovated'] > 0 and features_input['yr_renovated'] <= PREDICTION_YEAR : # Añadida condición lógica
            features_input['yrs_since_renovation'] = PREDICTION_YEAR - features_input['yr_renovated']
            features_input['was_renovated'] = 1
        else:
            features_input['yrs_since_renovation'] = features_input['age_at_sale'] # O 0, según tu lógica de entrenamiento
            features_input['was_renovated'] = 0
        
        input_df = pd.DataFrame([features_input])

        # Separar 'city' para OHE y las otras características
        city_to_encode_df = input_df[['city']]
        
        # Columnas numéricas y derivadas que NO son 'city' y NO son 'yr_renovated'
        numeric_and_derived_cols = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
            'yr_built', 'sale_month', 'age_at_sale', 'yrs_since_renovation', 
            'was_renovated'
        ]
        other_features_df = input_df[numeric_and_derived_cols]

        # One-Hot Encode 'city'
        city_encoded_array = city_encoder.transform(city_to_encode_df)
        # Usar las categorías del encoder para los nombres de las columnas, ya que get_feature_names_out
        # podría no estar disponible en versiones más antiguas de scikit-learn si el pickle es de allí.
        # Si city_encoder.categories_ está disponible y es la forma correcta:
        ohe_feature_names = [f"city_{cat}" for cat in city_encoder.categories_[0]]
        city_encoded_df = pd.DataFrame(city_encoded_array, columns=ohe_feature_names, index=other_features_df.index)

        # Combinar
        processed_df_unordered = pd.concat([other_features_df, city_encoded_df], axis=1)
        
        # Crear un DataFrame con todas las columnas esperadas por el modelo, en el orden correcto,
        # inicializadas con un tipo que permita .fillna(0.0) sin problemas.
        # Y asegurar que el tipo final sea float para el scaler.
        final_input_features_dict = {col: [0.0] for col in model_columns} # Inicializar con un valor float
        final_input_features = pd.DataFrame(final_input_features_dict, index=[0])

        # Llenar con los valores de processed_df_unordered, las columnas que no estén se quedarán como 0.0
        for col in processed_df_unordered.columns:
            if col in final_input_features.columns:
                final_input_features[col] = processed_df_unordered[col].values[0] # Asignar valor, no la Serie

        final_input_features = final_input_features[model_columns].astype(float) # Asegurar orden y tipo


        # Escalar las características
        scaled_features_array = scaler.transform(final_input_features)

        # Realizar la predicción
        prediction_log = model.predict(scaled_features_array)
        
        # Revertir la transformación logarítmica
        if TARGET_COLUMN_NAME == 'price_log':
            prediction_original_scale = np.expm1(prediction_log[0])
        else:
            prediction_original_scale = prediction_log[0]

        return render_template('index.html', 
                               prediction_text=f'Precio Estimado: ${prediction_original_scale:,.2f}',
                               city_options=sorted(city_options),
                               form_data=form_data,
                               PREDICTION_YEAR=PREDICTION_YEAR)

    except Exception as e:
        print(f"Error en la ruta /predict: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', 
                               prediction_text=f'Error durante la predicción. Por favor, verifica los datos ingresados y los logs del servidor.', 
                               city_options=sorted(city_options), 
                               PREDICTION_YEAR=PREDICTION_YEAR,
                               form_data=request.form.to_dict())

if __name__ == "__main__":
    # app.run(debug=True) # debug=True es bueno para desarrollo
    # Para despliegue en algunas plataformas o si debug=True causa problemas con recarga de modelos:
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))