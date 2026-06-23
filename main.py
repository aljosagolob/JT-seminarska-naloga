import subprocess
import sys


def load_dataset():
    subprocess.run([sys.executable, "dataset.py"])


def build_model():
    from model import DiarizationModel
    print("\nASR options:")
    print("  1. Whisper (multilingual)")
    print("  2. Slovenian (samolego/whisper-small-slovenian)")
    choice = input("Choose ASR [1/2]: ").strip()
    asr = "slovenian" if choice == "2" else "whisper"
    model = DiarizationModel(asr=asr)
    print(f"Model ready (asr={asr}).")
    return model


def benchmark():
    subprocess.run([sys.executable, "benchmark_files.py"])


def diarize(model):
    audio_file = input("Audio file path: ").strip()
    output_path = input("Output path [output/transcript.txt]: ").strip() or "output/transcript.txt"
    model.execPipeline(audio_file, output_path)


def optimize():
    subprocess.run([sys.executable, "optimize_preprocessing.py"])


def main():
    model = None

    while True:
        print("\n=== Speaker Diarization System ===")
        print("1. Load dataset to disk")
        print("2. Build model")
        print("3. Benchmark model")
        print("4. Diarize audio file")
        print("5. Optimize preprocessing params")
        print("0. Exit")

        choice = input("\nChoose option: ").strip()

        match choice:
            case "1":
                load_dataset()
            case "2":
                model = build_model()
            case "3":
                benchmark()
            case "4":
                if model is None:
                    print("Build a model first (option 2).")
                else:
                    diarize(model)
            case "5":
                optimize()
            case "0":
                break
            case _:
                print("Invalid option.")


if __name__ == "__main__":
    main()
