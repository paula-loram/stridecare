from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_model():
    # Time-series input (angles)
    time_input = Input(shape=(10, 6000), name='time_series_input')
    x = LSTM(512, activation='tanh', return_sequences=True)(time_input)
    x = LSTM(256, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)

    # Metadata input
    meta_input = Input(shape=(4,), name='meta_input')
    y = Dense(32, activation='relu')(meta_input)
    y = Dense(16, activation='relu')(y)

    # Combine both
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    z = Dense(6, activation='softmax')(z)

    model = Model(inputs=[time_input, meta_input], outputs=z)
    return model


from google.cloud import storage
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")


download_blob(
    bucket_name='stridecare-models',
    source_blob_name='models/my_model_weights.weights.h5',
    destination_file_name='local_model_weights.h5'
)
