import numpy as np
import pickle

from tensorflow.keras.models import load_model
from music21 import note, stream, chord

# Load model
model = load_model("music_model.keras")

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Load mapping
with open("mapping.pkl", "rb") as f:
    pitchnames = pickle.load(f)

note_to_int = {
    note_name: number
    for number, note_name in enumerate(pitchnames)
}

int_to_note = {
    number: note_name
    for number, note_name in enumerate(pitchnames)
}

sequence_length = 50

# Select random starting sequence
start = np.random.randint(
    0,
    len(notes) - sequence_length
)

pattern = [
    note_to_int[n]
    for n in notes[start:start + sequence_length]
]

output_notes = []

# Generate 200 notes
for _ in range(200):

    x = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    x = x / float(len(pitchnames))

    prediction = model.predict(
        x,
        verbose=0
    )

    index = np.argmax(prediction)

    result = int_to_note[index]

    output_notes.append(result)

    pattern.append(index)
    pattern = pattern[1:]

# Create MIDI
midi_stream = stream.Stream()

for pattern in output_notes:

    # Chord
    if "." in pattern:

        notes_in_chord = pattern.split(".")

        chord_notes = [
            note.Note(int(n))
            for n in notes_in_chord
        ]

        new_chord = chord.Chord(chord_notes)

        midi_stream.append(new_chord)

    # Single note
    else:

        new_note = note.Note(pattern)

        midi_stream.append(new_note)

# Save MIDI
midi_stream.write(
    "midi",
    fp="generated_music.mid"
)

print("Music generated successfully!")
print("Saved as: generated_music.mid")
