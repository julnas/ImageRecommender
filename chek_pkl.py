import pickle


def load_and_check_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    print(f"\n✅ Loaded {len(data)} entries from {pickle_path}\n")

    # Zeige die ersten 3 Einträge beispielhaft
    for i, (image_id, metrics) in enumerate(data.items()):
        print(f"Image ID: {image_id}")
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"  {key}: [list with {len(value)} values]")
            else:
                print(f"  {key}: {value}")
        print("-" * 40)

        if i >= 2:  # nur die ersten 3 anzeigen
            break


if __name__ == "__main__":
    load_and_check_pickle("image_metrics.pkl")
