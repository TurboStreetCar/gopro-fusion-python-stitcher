import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os

# --- PASTE YOUR EXACT SCRIPT IMPORTS HERE ---
import cv2
import numpy as np
import argparse
import time
import subprocess

# --- REDIRECT STDOUT TO THE GUI TEXTBOX ---
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str_output):
        self.text_widget.insert(tk.END, str_output)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks() # Forces UI refresh for \r progress bars

    def flush(self):
        pass

# --- YOUR EXACT SCRIPT FUNCTIONS ---
def load_pgm_map(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header not in ['P2', 'P5']:
            raise ValueError(f"Unsupported PGM format: {header}. Must be P2 or P5.")
        line = f.readline().decode('utf-8').strip()
        while line.startswith('#'):
            line = f.readline().decode('utf-8').strip()
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('utf-8').strip())
        if header == 'P2':
            data_array = np.fromfile(f, dtype=np.int32, sep=' ').reshape((height, width))
        else:
            raw_data = f.read()
            data_array = np.frombuffer(raw_data, dtype=np.int16).reshape((height, width))
        return data_array.astype(np.float32)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- MAIN GUI CLASS ---
class StitcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GoPro Fusion Direct-to-MOV Stitcher Engine")
        self.root.geometry("680x620")

        # Internal file storage lists
        self.videos = []

        self.maps = [
            "./pgm-examples/video-5_2k/fusion2sphere0_x.pgm",
            "./pgm-examples/video-5_2k/fusion2sphere1_x.pgm",
            "./pgm-examples/video-5_2k/fusion2sphere0_y.pgm",
            "./pgm-examples/video-5_2k/fusion2sphere1_y.pgm"
        ]

        self.masks = [
            "./pgm-examples/video-5_2k/frontmask.png",
            "./pgm-examples/video-5_2k/backmask.png"
        ]

        self.output_file = tk.StringVar()

        # UI Layout Construction
        self.create_widgets()

        # Redirect standard console prints directly into GUI text box
        sys.stdout = TextRedirector(self.output_text)
        sys.stderr = TextRedirector(self.output_text)

    def create_widgets(self):
        # Group 1: Video Selector
        tk.Label(self.root, text="1. Front Video:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.front_entry = tk.Entry(self.root, width=50, state="readonly")
        self.front_entry.grid(row=0, column=1, padx=5)
        tk.Button(self.root, text="Select Front Video", command=self.browse_front_video).grid(row=0, column=2, padx=5)

        tk.Label(self.root, text="2. Rear Video:", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.rear_entry = tk.Entry(self.root, width=50, state="readonly")
        self.rear_entry.grid(row=1, column=1, padx=5)
        tk.Button(self.root, text="Select Rear Video", command=self.browse_rear_video).grid(row=1, column=2, padx=5)

        # # Group 2: Maps Selector
        # tk.Label(self.root, text="2. Mapping Vectors:", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, padx=10, pady=10, sticky="w")
        # self.map_entry = tk.Entry(self.root, width=50, state="readonly")
        # self.map_entry.grid(row=1, column=1, padx=5)
        # tk.Button(self.root, text="Select 4 PGM Maps", command=self.browse_maps).grid(row=1, column=2, padx=5)
        #
        # # Group 3: Masks Selector
        # tk.Label(self.root, text="3. Blend Masks:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, padx=10, pady=10, sticky="w")
        # self.mask_entry = tk.Entry(self.root, width=50, state="readonly")
        # self.mask_entry.grid(row=2, column=1, padx=5)
        # tk.Button(self.root, text="Select 2 Mask PNGs", command=self.browse_masks).grid(row=2, column=2, padx=5)

        # Output Target Selector
        tk.Label(self.root, text="Output Target:", font=('Helvetica', 10, 'bold')).grid(row=3, column=0, padx=10, pady=15, sticky="w")
        tk.Entry(self.root, textvariable=self.output_file, width=50, state="readonly").grid(row=3, column=1, padx=5)
        tk.Button(self.root, text="Save As MOV", command=self.browse_output).grid(row=3, column=2, padx=5)

        # Run Button & Progress Bar
        self.run_btn = tk.Button(self.root, text="🚀 START STITCHING PROCESS", font=("Helvetica", 11, "bold"), bg="#4CAF50", fg="white", command=self.start_thread, height=2)
        self.run_btn.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew", padx=15)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, padx=15, pady=5, sticky="ew")

        # Multiline Terminal Log Box
        tk.Label(self.root, text="Console Output Execution Log:", font=('Helvetica', 9, 'normal')).grid(row=6, column=0, padx=15, sticky="w")
        self.txt_frame = tk.Frame(self.root)
        self.txt_frame.grid(row=7, column=0, columnspan=3, padx=15, pady=5, sticky="nsew")

        self.scrollbar = tk.Scrollbar(self.txt_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(self.txt_frame, height=14, width=78, yscrollcommand=self.scrollbar.set, bg="#1e1e1e", fg="#ffffff", insertbackground="white", font=("Courier", 9))
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.output_text.yview)

        # Configure dynamic layout weights
        self.root.grid_rowconfigure(7, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    # --- FILE SELECTOR LOGIC ---
    def browse_front_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Front Video",
            initialdir="/run/media/dan/870EVO/Pictures/GoPro Fusion/Projects",
            filetypes=[
                ("Video Files", "*.mp4 *.MP4 *.mov *.MOV"),
                ("All Files", "*")
            ]
        )

        if file_path:
            self.front_video = file_path

            self.update_entry_text(
                self.front_entry,
                os.path.basename(file_path)
            )


    def browse_rear_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Rear Video",
            initialdir="/run/media/dan/870EVO/Pictures/GoPro Fusion/Projects",
            filetypes=[
                ("Video Files", "*.mp4 *.MP4 *.mov *.MOV"),
                ("All Files", "*")
            ]
        )

        if file_path:
            self.rear_video = file_path

            self.update_entry_text(
                self.rear_entry,
                os.path.basename(file_path)
            )

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(title="Save Production Master", defaultextension=".mov", filetypes=[("MOV Video", "*.mov")])
        if file_path:
            self.output_file.set(file_path)

    def update_entry_text(self, entry_widget, text):
        entry_widget.config(state="normal")
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, text)
        entry_widget.config(state="readonly")

    # --- MULTI-THREAD CONTROL ---
    def start_thread(self):
        """Runs processing on a background thread so the GUI frame does not freeze or crash."""
        if not (
            self.front_video and
            self.rear_video and
            self.output_file.get()
        ):
            messagebox.showwarning("Missing Paths", "Please complete steps 1, 2, 3, and choose an output path.")
            return

        self.run_btn.config(state="disabled", text="PROCESSING VIDEO STREAMS...")
        self.output_text.delete(1.0, tk.END)

        # Execute processing thread
        threading.Thread(target=self.core_processing_engine, daemon=True).start()

    def seam_exposure_preprocess(self, fr, bk):
        fr_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bk_gray = cv2.cvtColor(bk, cv2.COLOR_BGR2GRAY).astype(np.float32)

        fr_mean = np.mean(fr_gray)
        bk_mean = np.mean(bk_gray)

        if bk_mean < 1e-6:
            return fr, bk

        gain = fr_mean / bk_mean
        bk = cv2.convertScaleAbs(bk, alpha=gain)

        return fr, bk

    def compute_luma(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- CORE STITCHING ENGINE CORE INTEGRATION ---
    def core_processing_engine(self):
        # Initialize temp path early so the finally block can safely evaluate it
        temp_silent_path = ""
        try:
            # Map paths to variables out of sorted lists
            front_video = self.front_video
            back_video = self.rear_video
            xmap0, xmap1, ymap0, ymap1 = self.maps[0], self.maps[1], self.maps[2], self.maps[3]
            mask0_path, mask1_path = self.masks[0], self.masks[1]
            output_mov = self.output_file.get()
            fps_val = 30.0

            cap_fr = cv2.VideoCapture(front_video)
            cap_bk = cv2.VideoCapture(back_video)

            total_frames = min(int(cap_fr.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_bk.get(cv2.CAP_PROP_FRAME_COUNT)))
            if total_frames <= 0:
                print("Error: Could not read video streams. Verify your input paths.", file=sys.stderr)
                return

            print(f"Successfully loaded video streams ({total_frames} frames).")
            print("Loading map vectors and normalizing mask arrays...")

            map0_x = load_pgm_map(xmap0)
            map0_y = load_pgm_map(ymap0)
            map1_x = load_pgm_map(xmap1)
            map1_y = load_pgm_map(ymap1)

            mask0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)

            mask0_3ch = cv2.merge([mask0, mask0, mask0])
            mask1_3ch = cv2.merge([mask1, mask1, mask1])

            map_height, map_width = map0_x.shape
            print(f"Targeting absolute native 360 dimensions: {map_width}x{map_height}")

            base_dir = os.path.dirname(os.path.abspath(output_mov))
            os.makedirs(base_dir, exist_ok=True)
            temp_silent_path = os.path.join(base_dir, ".temp_silent.mov")

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_video = cv2.VideoWriter(temp_silent_path, fourcc, fps_val, (map_width, map_height))

            if not out_video.isOpened():
                print("Error: Failed to initialize the video writer stream.", file=sys.stderr)
                sys.exit(1)

            umat_map0_x = cv2.UMat(map0_x)
            umat_map0_y = cv2.UMat(map0_y)
            umat_map1_x = cv2.UMat(map1_x)
            umat_map1_y = cv2.UMat(map1_y)

            umat_mask0 = cv2.UMat(mask0_3ch)
            umat_mask1 = cv2.UMat(mask1_3ch)

            print("Rendering temporary silent video layout to buffer space...")
            print("Executing GPU-optimized remap engine...")

            frame_idx = 1
            start_time = time.time()
            last_check_time = start_time
            last_check_frame = 0
            interval = 30

            mask0 = cv2.GaussianBlur(mask0, (0, 0), 15)
            mask1 = cv2.GaussianBlur(mask1, (0, 0), 15)
            mask0 = cv2.merge([mask0, mask0, mask0])
            mask1 = cv2.merge([mask1, mask1, mask1])

            while True:
                ret_fr, img_fr = cap_fr.read()
                ret_bk, img_bk = cap_bk.read()

                if not ret_fr or not ret_bk:
                    break

                img_fr, img_bk = self.seam_exposure_preprocess(img_fr, img_bk)

                umat_fr = cv2.UMat(img_fr)
                umat_bk = cv2.UMat(img_bk)



                stitched_fr = cv2.remap(
                    umat_fr,
                    umat_map0_x,
                    umat_map0_y,
                    interpolation=cv2.INTER_LINEAR
                )

                stitched_bk = cv2.remap(
                    umat_bk,
                    umat_map1_x,
                    umat_map1_y,
                    interpolation=cv2.INTER_LINEAR
                )

                scale = 0.25

                small_fr = cv2.resize(stitched_fr.get(), None, fx=scale, fy=scale)
                small_bk = cv2.resize(stitched_bk.get(), None, fx=scale, fy=scale)

                #stitched_fr = stitched_fr.get()
                #stitched_bk = stitched_bk.get()

                lum_fr = cv2.cvtColor(small_fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                lum_bk = cv2.cvtColor(small_bk, cv2.COLOR_BGR2GRAY).astype(np.float32)

                gain_map_small = lum_fr / (lum_bk + 1e-6)
                gain_map_small = cv2.GaussianBlur(gain_map_small, (0,0), 10)

                gain_map = cv2.resize(gain_map_small, (map_width, map_height))
                gain_map = np.clip(gain_map, 0.5, 2.0)

                stitched_bk = stitched_bk.astype(np.float32)
                stitched_bk *= gain_map[..., None]
                stitched_bk = np.clip(stitched_bk, 0, 255).astype(np.uint8)

                blended_fr = stitched_fr.astype(np.float32) * (mask0 / 255.0)
                blended_bk = stitched_bk.astype(np.float32) * (mask1 / 255.0)

                #final_frame = cv2.add(blended_fr, blended_bk).astype(np.uint8)
                final_frame = cv2.add(blended_fr, blended_bk)

                final_frame_resized = cv2.resize(
                    final_frame,
                    (map_width, map_height)
                )

                out_video.write(final_frame_resized)

                if frame_idx % interval == 0 or frame_idx == total_frames:
                    current_time = time.time()

                    elapsed_interval_time = (
                        current_time - last_check_time
                    )

                    frames_processed_in_interval = (
                        frame_idx - last_check_frame
                    )

                    current_fps = (
                        frames_processed_in_interval /
                        elapsed_interval_time
                        if elapsed_interval_time > 0 else 0.0
                    )

                    total_elapsed_time = current_time - start_time

                    avg_fps = (
                        frame_idx / total_elapsed_time
                        if total_elapsed_time > 0 else 0.0
                    )

                    remaining_frames = total_frames - frame_idx

                    eta_seconds = (
                        remaining_frames / avg_fps
                        if avg_fps > 0 else 0.0
                    )

                    elapsed_str = format_time(total_elapsed_time)
                    eta_str = format_time(eta_seconds)

                    percent_done = (
                        frame_idx / total_frames
                    ) * 100

                    self.progress["value"] = percent_done

                    sys.stdout.write(
                        f"Progress: {frame_idx}/{total_frames} "
                        f"[{percent_done:.1f}%] | "
                        f"Speed: {current_fps:.2f} fps | "
                        f"Elapsed: {elapsed_str} | "
                        f"ETA: {eta_str}\n"
                    )

                    sys.stdout.flush()

                    last_check_time = current_time
                    last_check_frame = frame_idx

                frame_idx += 1

            cap_fr.release()
            cap_bk.release()
            out_video.release()

            print(
                "\n\nSuccess! High-performance video compilation completed."
            )

            # ----------------------------------------------------
            # AUTOMATED AUDIO MERGE PASS
            # ----------------------------------------------------

            print(
                "\nInitializing automated FFmpeg spatial audio pass..."
            )

            print(
                f"Downmixing tracks to standard "
                f"Resolve-supported Stereo -> {output_mov}"
            )

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "warning",
                "-i",
                temp_silent_path,
                "-i",
                front_video,
                "-i",
                back_video,
                "-filter_complex",
                "[1:a][2:a]amix=inputs=2:"
                "duration=first:"
                "dropout_transition=0[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-ac",
                "2",
                output_mov
            ]

            subprocess.run(
                ffmpeg_cmd,
                check=True
            )

            print(
                "\nSuccess! Multi-channel sound matrix compiled."
            )

            print(
                "Your master edit file is ready for "
                f"DaVinci Resolve at:\n -> {output_mov}"
            )

            messagebox.showinfo(
                "Process Complete",
                "Video stitched and audio merged successfully!"
            )

        except subprocess.CalledProcessError as e:
            print(
                f"\nError: Background FFmpeg execution failed: {e}",
                file=sys.stderr
            )

            messagebox.showerror(
                "FFmpeg Error",
                f"Background FFmpeg execution failed:\n{e}"
            )

        except Exception as e:
            print(
                f"\nExecution Failed unexpectedly: {str(e)}",
                file=sys.stderr
            )

            messagebox.showerror(
                "Error",
                f"An error occurred:\n{str(e)}"
            )

        finally:
            # Cleanup temp file
            if (
                temp_silent_path and
                os.path.exists(temp_silent_path)
            ):
                os.remove(temp_silent_path)
                print(
                    "Cleaned up intermediate temporary file artifacts."
                )

            self.reset_ui()

    def reset_ui(self):
        self.run_btn.config(
            state="normal",
            text="🚀 START STITCHING PROCESS"
        )

        self.progress["value"] = 0


if __name__ == "__main__":
    root = tk.Tk()
    app = StitcherGUI(root)
    root.mainloop()
