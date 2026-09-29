from music21 import converter, instrument, note, chord
import glob
import pickle

notes = []

# Read all MIDI files
for file in glob.glob("dataset/*.mid"):

    print("Processing:", file)

    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        # Single note
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        # Chord
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

print("Total notes:", len(notes))

# Save extracted notes
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Preprocessing completed!")