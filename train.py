import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

print("Total notes:", len(notes))

# Unique notes
pitchnames = sorted(set(notes))

print("Unique notes:", len(pitchnames))

# Create mapping
note_to_int = {note: number for number, note in enumerate(pitchnames)}

sequence_length = 50

network_input = []
network_output = []

# Create sequences
for i in range(len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append(
        [note_to_int[n] for n in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_patterns = len(network_input)

print("Training sequences:", n_patterns)

# Reshape input
X = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

# Normalize
X = X / float(len(pitchnames))

# One-hot encoding
y = to_categorical(
    network_output,
    num_classes=len(pitchnames)
)

# Create LSTM model
model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(256)
)

model.add(Dropout(0.3))

model.add(
    Dense(len(pitchnames), activation="softmax")
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

# Train
model.fit(
    X,
    y,
    epochs=30,
    batch_size=64
)

# Save model
model.save("music_model.keras")

# Save note mapping
with open("mapping.pkl", "wb") as f:
    pickle.dump(pitchnames, f)

print("Training completed!")