
import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
import subprocess
import threading
import sys

PROJECT_DIR = Path(__file__).resolve().parent

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class LightHiveUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("LightHive | AI Stage Lighting")
        self.geometry("850x650")
        self.minsize(700, 550)

        self.song_path = None

        ctk.CTkLabel(
            self,
            text="LIGHTHIVE",
            font=("Arial", 32, "bold")
        ).pack(pady=(25, 4))

        ctk.CTkLabel(
            self,
            text="AI-POWERED VIRTUAL STAGE LIGHTING",
            text_color="#A78BFA",
            font=("Arial", 13)
        ).pack(pady=(0, 25))

        self.upload_button = ctk.CTkButton(
            self,
            text="Choose Audio File",
            command=self.choose_audio,
            height=45,
            width=220
        )
        self.upload_button.pack(pady=8)

        self.song_label = ctk.CTkLabel(
            self,
            text="No song selected",
            wraplength=650
        )
        self.song_label.pack(pady=10)

        self.analyse_button = ctk.CTkButton(
            self,
            text="Analyse Song & Control Stage",
            command=self.analyse_song,
            height=45,
            width=280,
            fg_color="#6D28D9",
            hover_color="#5B21B6"
        )
        self.analyse_button.pack(pady=15)

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready — select a song to begin",
            text_color="#86EFAC"
        )
        self.status_label.pack(pady=5)

        self.output = ctk.CTkTextbox(
            self,
            width=760,
            height=300,
            font=("Consolas", 12)
        )
        self.output.pack(padx=25, pady=15, fill="both", expand=True)

    def choose_audio(self):
        filename = filedialog.askopenfilename(
            title="Choose a song",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg"),
                ("All files", "*.*")
            ]
        )

        if filename:
            self.song_path = Path(filename)
            self.song_label.configure(
                text=f"Selected: {self.song_path.name}"
            )
            self.status_label.configure(
                text="Song selected — ready to analyse"
            )

    def analyse_song(self):
        if self.song_path is None:
            self.status_label.configure(
                text="Please choose an audio file first",
                text_color="#FCA5A5"
            )
            return

        if not self.song_path.exists():
            self.status_label.configure(
                text="Audio file not found",
                text_color="#FCA5A5"
            )
            return

        self.analyse_button.configure(state="disabled")
        self.status_label.configure(
            text="Analysing audio and generating lighting...",
            text_color="#FDE68A"
        )
        self.output.delete("1.0", "end")
        self.output.insert("end", "Starting LightHive AI...\n")

        threading.Thread(
            target=self.run_pipeline,
            daemon=True
        ).start()

    def run_pipeline(self):
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_DIR / "lighthive_live.py"),
                    str(self.song_path)
                ],
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            for line in process.stdout:
                self.after(
                    0,
                    lambda line=line: self.output.insert("end", line)
                )
                self.after(
                    0,
                    lambda: self.output.see("end")
                )

            return_code = process.wait()

            if return_code == 0:
                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text="Analysis finished — check your Unity stage!",
                        text_color="#86EFAC"
                    )
                )
            else:
                self.after(
                    0,
                    lambda: self.status_label.configure(
                        text="Analysis failed — check the output below",
                        text_color="#FCA5A5"
                    )
                )

        except Exception as error:
            self.after(
                0,
                lambda: self.output.insert("end", f"\nERROR: {error}\n")
            )
            self.after(
                0,
                lambda: self.status_label.configure(
                    text="Could not start the AI pipeline",
                    text_color="#FCA5A5"
                )
            )

        finally:
            self.after(
                0,
                lambda: self.analyse_button.configure(state="normal")
            )


if __name__ == "__main__":
    app = LightHiveUI()
    app.mainloop()
