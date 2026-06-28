import os
import sys
import wave
import queue
import time
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox

# Fallback mechanism for RVC Inference dependency
try:
    from rvc_python.infer import RVCInference
    RVC_AVAILABLE = True
except BaseException:
    RVC_AVAILABLE = False


class AnonyVoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AnonyVox — AI Voice Morphing & Cryptographic Redaction Engine")
        self.root.geometry("980x700")
        self.root.minsize(900, 650)
        self.root.configure(bg="#0b0a12")  # Cyberpunk Dark Violet-Black

        # Create lock for RVC inference thread-safety
        self.rvc_lock = threading.Lock()
        
        # UI & Engine Variables
        self.engine_active = False
        self.scramble_active = False
        self.worker_running = False
        self.stream = None
        self.worker_thread = None
        
        # Audio Buffering & Processing Variables
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.accumulator = []
        self.stream_blocksize = 1024
        self.carrier_phase_accumulator = 0
        
        # Model Tracking
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            
        self.current_model_path = None
        self.current_index_path = None
        
        # Temporary files for batch RVC inference (stored in project root or system temp)
        self.temp_in_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_voice_in.wav")
        self.temp_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_voice_out.wav")
        
        # Thread-safe Visualizer Data
        self.visualizer_data = np.zeros(512, dtype=np.float32)
        self.visualizer_lock = threading.Lock()

        # Initialize core RVC inference engine
        self.rvc = None
        self.cuda_active = False
        self.initialize_rvc_engine()
        
        # Set up Tkinter variables
        self.pitch_val = tk.IntVar(value=0)
        self.spectral_inversion_val = tk.DoubleVar(value=0.0)
        self.ring_mod_val = tk.IntVar(value=0)
        self.tremolo_depth_val = tk.DoubleVar(value=0.0)
        self.distortion_val = tk.DoubleVar(value=1.0)
        self.monitor_audio_var = tk.BooleanVar(value=True)
        self.model_var = tk.StringVar()
        self.input_device_var = tk.StringVar()
        self.output_device_var = tk.StringVar()
        self.rvc_enabled_var = tk.BooleanVar(value=True if RVC_AVAILABLE else False)
        
        # Build UI Components
        self.setup_ui_styles()
        self.build_ui_layout()
        
        # Load Hardware & Model lists
        self.populate_devices()
        self.refresh_model_list()
        
        # Start Waveform animation loop
        self.draw_visualizer()

    def initialize_rvc_engine(self):
        if not RVC_AVAILABLE:
            print("Warning: rvc-python library is not installed.")
            return

        # Attempt GPU CUDA Initialization
        try:
            # Explicit CUDA:0 execution context
            self.rvc = RVCInference(device="cuda:0")
            self.cuda_active = True
            print("Successfully initialized RVCInference on CUDA:0 GPU.")
        except Exception as e:
            print(f"CUDA initialization failed: {e}. Attempting CPU fallback...")
            try:
                # CPU Fallback
                self.rvc = RVCInference(device="cpu")
                self.cuda_active = False
                print("Successfully initialized RVCInference on CPU.")
            except Exception as ex:
                self.rvc = None
                self.cuda_active = False
                print(f"RVC Engine initialization failed entirely: {ex}")

    def setup_ui_styles(self):
        # Configure Custom Combobox Style matching Dark Synthwave theme
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure(
            "TCombobox",
            fieldbackground="#1c1a2e",
            background="#0b0a12",
            foreground="#ffffff",
            bordercolor="#00f0ff",
            lightcolor="#00f0ff",
            darkcolor="#0b0a12",
            arrowcolor="#00f0ff"
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[('readonly', '#1c1a2e'), ('disabled', '#0b0a12')],
            foreground=[('readonly', '#ffffff'), ('disabled', '#64748b')]
        )

    def build_ui_layout(self):
        # Master Container Grid configuration
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        # ----------------------------------------------------
        # 1. Header Frame
        # ----------------------------------------------------
        header_frame = tk.Frame(self.root, bg="#0b0a12", height=80)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        header_frame.grid_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="ANONYVOX",
            font=("Segoe UI", 24, "bold"),
            fg="#ff007f",  # Neon Pink
            bg="#0b0a12"
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            header_frame,
            text="AI-POWERED REAL-TIME VOICE MORPHING & CRYPTOGRAPHIC BLUR ENGINE",
            font=("Segoe UI", 9, "bold"),
            fg="#00f0ff",  # Neon Cyan
            bg="#0b0a12"
        )
        subtitle_label.pack(anchor="w")

        # ----------------------------------------------------
        # 2. Left Column Control Panel
        # ----------------------------------------------------
        left_panel = tk.Frame(self.root, bg="#0b0a12")
        left_panel.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        left_panel.columnconfigure(0, weight=1)

        # Module A: Audio Routing / Hardware Hooks
        hw_frame = tk.LabelFrame(
            left_panel, 
            text=" CORE HARDWARE ROUTING ", 
            font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#00f0ff"
        )
        hw_frame.pack(fill="x", pady=10, ipady=5)
        hw_frame.columnconfigure(1, weight=1)

        tk.Label(hw_frame, text="Input Microphone:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.input_dropdown = ttk.Combobox(hw_frame, textvariable=self.input_device_var, state="readonly")
        self.input_dropdown.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.input_dropdown.bind("<<ComboboxSelected>>", self.on_device_changed)

        tk.Label(hw_frame, text="Output Speaker/VAC:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.output_dropdown = ttk.Combobox(hw_frame, textvariable=self.output_device_var, state="readonly")
        self.output_dropdown.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        self.output_dropdown.bind("<<ComboboxSelected>>", self.on_device_changed)

        # Monitor Output Checkbox
        self.monitor_chk = tk.Checkbutton(
            hw_frame, 
            text="Monitor Output (Hear Self / Echo)", 
            variable=self.monitor_audio_var,
            onvalue=True, offvalue=False,
            font=("Segoe UI", 9, "bold"),
            fg="#00f0ff", bg="#13121f", selectcolor="#13121f",
            activebackground="#13121f", activeforeground="#00f0ff"
        )
        self.monitor_chk.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Refresh Hardware Selector Button
        refresh_btn = tk.Button(
            hw_frame, text="REFRESH HARDWARE",
            bg="#1c1a2e", fg="#00f0ff", activebackground="#2d2a4a", activeforeground="#ffffff",
            font=("Segoe UI", 8, "bold"), bd=1, relief="solid", highlightbackground="#00f0ff",
            command=self.populate_devices, padx=10, pady=4
        )
        refresh_btn.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        refresh_btn.bind("<Enter>", lambda e: refresh_btn.config(bg="#2d2a4a"))
        refresh_btn.bind("<Leave>", lambda e: refresh_btn.config(bg="#1c1a2e"))

        # Module B: RVC AI Engine Setup
        rvc_frame = tk.LabelFrame(
            left_panel, 
            text=" AI RVC MODULATOR ", 
            font=("Segoe UI", 10, "bold"),
            fg="#ff007f", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#ff007f"
        )
        rvc_frame.pack(fill="x", pady=10, ipady=5)
        rvc_frame.columnconfigure(1, weight=1)

        # Enable/Disable RVC
        self.rvc_chk = tk.Checkbutton(
            rvc_frame, 
            text="Activate AI Voice Morphing Pipeline", 
            variable=self.rvc_enabled_var,
            onvalue=True, offvalue=False,
            font=("Segoe UI", 9, "bold"),
            fg="#ff007f", bg="#13121f", selectcolor="#13121f",
            activebackground="#13121f", activeforeground="#ff007f",
            command=self.on_rvc_toggle
        )
        self.rvc_chk.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        tk.Label(rvc_frame, text="Select RVC Model:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.model_dropdown = ttk.Combobox(rvc_frame, textvariable=self.model_var, state="readonly")
        self.model_dropdown.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Pitch scale slider component (-12 to +12 semitones)
        tk.Label(rvc_frame, text="Pitch Transposition:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.pitch_slider_frame = tk.Frame(rvc_frame, bg="#13121f")
        self.pitch_slider_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.pitch_slider_frame.columnconfigure(0, weight=1)

        self.pitch_slider = tk.Scale(
            self.pitch_slider_frame, from_=-12, to=12, orient="horizontal",
            variable=self.pitch_val, showvalue=False,
            bg="#13121f", fg="#ffffff", troughcolor="#1c1a2e",
            activebackground="#ff007f", highlightthickness=0, bd=0
        )
        self.pitch_slider.grid(row=0, column=0, sticky="ew")
        
        self.pitch_lbl = tk.Label(self.pitch_slider_frame, text="0 st", font=("Segoe UI", 9, "bold"), fg="#ff007f", bg="#13121f", width=6)
        self.pitch_lbl.grid(row=0, column=1, padx=5)
        self.pitch_val.trace_add("write", lambda *args: self.pitch_lbl.config(text=f"{self.pitch_val.get():+d} st"))

        # Voice Profile Presets Dropdown
        tk.Label(rvc_frame, text="Vocal Preset:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.preset_var = tk.StringVar(value="Select Preset...")
        self.preset_dropdown = ttk.Combobox(rvc_frame, textvariable=self.preset_var, values=["Select Preset...", "Deep Male", "High Female", "Astral Echo", "Overdrive Cyborg", "Default Reset"], state="readonly")
        self.preset_dropdown.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        self.preset_dropdown.bind("<<ComboboxSelected>>", self.on_preset_selected)

        # Module C: DSP Voice Modifiers Frame
        si_frame = tk.LabelFrame(
            left_panel, 
            text=" DSP EFFECTS MATRIX ", 
            font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#00f0ff"
        )
        si_frame.pack(fill="x", pady=10, ipady=5)
        si_frame.columnconfigure(1, weight=1)

        # Effect 1: Spectral Inversion
        tk.Label(si_frame, text="Spectral Inversion:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.si_slider_frame = tk.Frame(si_frame, bg="#13121f")
        self.si_slider_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.si_slider_frame.columnconfigure(0, weight=1)
        self.si_slider = tk.Scale(
            self.si_slider_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
            variable=self.spectral_inversion_val, showvalue=False,
            bg="#13121f", fg="#ffffff", troughcolor="#1c1a2e",
            activebackground="#00f0ff", highlightthickness=0, bd=0
        )
        self.si_slider.grid(row=0, column=0, sticky="ew")
        self.si_lbl = tk.Label(self.si_slider_frame, text="0.00", font=("Segoe UI", 9, "bold"), fg="#00f0ff", bg="#13121f", width=6)
        self.si_lbl.grid(row=0, column=1, padx=5)
        self.spectral_inversion_val.trace_add("write", lambda *args: self.si_lbl.config(text=f"{self.spectral_inversion_val.get():.2f}"))

        # Effect 2: Ring Modulation
        tk.Label(si_frame, text="Ring Modulator:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.ring_slider_frame = tk.Frame(si_frame, bg="#13121f")
        self.ring_slider_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        self.ring_slider_frame.columnconfigure(0, weight=1)
        self.ring_slider = tk.Scale(
            self.ring_slider_frame, from_=0, to=1200, resolution=10, orient="horizontal",
            variable=self.ring_mod_val, showvalue=False,
            bg="#13121f", fg="#ffffff", troughcolor="#1c1a2e",
            activebackground="#00f0ff", highlightthickness=0, bd=0
        )
        self.ring_slider.grid(row=0, column=0, sticky="ew")
        self.ring_lbl = tk.Label(self.ring_slider_frame, text="0 Hz", font=("Segoe UI", 9, "bold"), fg="#00f0ff", bg="#13121f", width=6)
        self.ring_lbl.grid(row=0, column=1, padx=5)
        self.ring_mod_val.trace_add("write", lambda *args: self.ring_lbl.config(text=f"{self.ring_mod_val.get()} Hz"))

        # Effect 3: Tremolo
        tk.Label(si_frame, text="Vocal Tremolo:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.trem_slider_frame = tk.Frame(si_frame, bg="#13121f")
        self.trem_slider_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
        self.trem_slider_frame.columnconfigure(0, weight=1)
        self.trem_slider = tk.Scale(
            self.trem_slider_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
            variable=self.tremolo_depth_val, showvalue=False,
            bg="#13121f", fg="#ffffff", troughcolor="#1c1a2e",
            activebackground="#00f0ff", highlightthickness=0, bd=0
        )
        self.trem_slider.grid(row=0, column=0, sticky="ew")
        self.trem_lbl = tk.Label(self.trem_slider_frame, text="0.00", font=("Segoe UI", 9, "bold"), fg="#00f0ff", bg="#13121f", width=6)
        self.trem_lbl.grid(row=0, column=1, padx=5)
        self.tremolo_depth_val.trace_add("write", lambda *args: self.trem_lbl.config(text=f"{self.tremolo_depth_val.get():.2f}"))

        # Effect 4: Vocal Distortion
        tk.Label(si_frame, text="Overdrive Dist:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#13121f").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.dist_slider_frame = tk.Frame(si_frame, bg="#13121f")
        self.dist_slider_frame.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        self.dist_slider_frame.columnconfigure(0, weight=1)
        self.dist_slider = tk.Scale(
            self.dist_slider_frame, from_=1.0, to=10.0, resolution=0.1, orient="horizontal",
            variable=self.distortion_val, showvalue=False,
            bg="#13121f", fg="#ffffff", troughcolor="#1c1a2e",
            activebackground="#00f0ff", highlightthickness=0, bd=0
        )
        self.dist_slider.grid(row=0, column=0, sticky="ew")
        self.dist_lbl = tk.Label(self.dist_slider_frame, text="1.0x", font=("Segoe UI", 9, "bold"), fg="#00f0ff", bg="#13121f", width=6)
        self.dist_lbl.grid(row=0, column=1, padx=5)
        self.distortion_val.trace_add("write", lambda *args: self.dist_lbl.config(text=f"{self.distortion_val.get():.1f}x"))

        # Engine Power Switch
        self.start_btn = tk.Button(
            left_panel, text="START ENGINE",
            bg="#00f0ff", fg="#0b0a12", activebackground="#80f8ff", activeforeground="#0b0a12",
            font=("Segoe UI", 12, "bold"), bd=0, cursor="hand2",
            command=self.toggle_engine, pady=10
        )
        self.start_btn.pack(fill="x", pady=15)
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#80f8ff") if not self.engine_active else None)
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg="#00f0ff") if not self.engine_active else None)

        # ----------------------------------------------------
        # 3. Right Column Output & Diagnostics Panel
        # ----------------------------------------------------
        right_panel = tk.Frame(self.root, bg="#0b0a12")
        right_panel.grid(row=1, column=1, sticky="nsew", padx=20, pady=10)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        # Module D: AI Semantic Redaction ("Hold to Scramble" button)
        redact_frame = tk.LabelFrame(
            right_panel, 
            text=" SEMANTIC REDACTION ", 
            font=("Segoe UI", 10, "bold"),
            fg="#ff007f", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#ff007f"
        )
        redact_frame.pack(fill="x", pady=10, ipady=5)
        
        self.scramble_btn = tk.Button(
            redact_frame, text="HOLD TO SCRAMBLE VOICE",
            bg="#1c1a2e", fg="#ff2a5f", activebackground="#ff2a5f", activeforeground="#ffffff",
            font=("Segoe UI", 12, "bold"), bd=1, relief="solid", highlightbackground="#ff2a5f",
            cursor="hand2", pady=15
        )
        self.scramble_btn.pack(fill="x", padx=15, pady=15)
        
        # Mouse listener bindings for click-and-hold scrambling functionality
        self.scramble_btn.bind("<ButtonPress-1>", self.on_scramble_start)
        self.scramble_btn.bind("<ButtonRelease-1>", self.on_scramble_stop)
        self.scramble_btn.bind("<Enter>", lambda e: self.scramble_btn.config(bg="#3a1324") if not self.scramble_active else None)
        self.scramble_btn.bind("<Leave>", lambda e: self.scramble_btn.config(bg="#1c1a2e") if not self.scramble_active else None)

        # Module G: Utilities and Virtual Audio Cable Router
        util_frame = tk.LabelFrame(
            right_panel, 
            text=" UTILITIES & ROUTING ", 
            font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#00f0ff"
        )
        util_frame.pack(fill="x", pady=10, ipady=5)
        util_frame.columnconfigure(0, weight=1)
        util_frame.columnconfigure(1, weight=1)
        
        self.is_recording = False
        self.recorded_blocks = []
        
        self.record_btn = tk.Button(
            util_frame, text="🔴 START RECORDING",
            bg="#1c1a2e", fg="#ff2a5f", activebackground="#ff2a5f", activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), bd=1, relief="solid", highlightbackground="#ff2a5f",
            cursor="hand2", command=self.toggle_recording, pady=6
        )
        self.record_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.route_btn = tk.Button(
            util_frame, text="🔌 AUTO-ROUTE VIRTUAL MIC",
            bg="#1c1a2e", fg="#00f0ff", activebackground="#00f0ff", activeforeground="#0b0a12",
            font=("Segoe UI", 9, "bold"), bd=1, relief="solid", highlightbackground="#00f0ff",
            cursor="hand2", command=self.auto_route_virtual_mic, pady=6
        )
        self.route_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Module E: Live Visualizer Screen
        viz_frame = tk.LabelFrame(
            right_panel, 
            text=" DETECTOR WAVEFORM ", 
            font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#00f0ff"
        )
        viz_frame.pack(fill="both", expand=True, pady=10)
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(viz_frame, bg="#06070d", bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Module F: Hardware/System Status Logs
        log_frame = tk.LabelFrame(
            right_panel, 
            text=" SYSTEM ENGINE LOGS ", 
            font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#13121f", 
            bd=1, relief="solid", highlightbackground="#00f0ff"
        )
        log_frame.pack(fill="x", pady=10)

        self.status_label = tk.Label(
            log_frame, 
            text="System Initialized. Idle.", 
            font=("Segoe UI", 9, "bold"),
            fg="#a78bfa", bg="#13121f", anchor="w"
        )
        self.status_label.pack(fill="x", padx=10, pady=5)

        self.log_text = tk.Text(
            log_frame, height=5, font=("Courier New", 8),
            bg="#06070d", fg="#a78bfa", bd=0, highlightthickness=0
        )
        self.log_text.pack(fill="x", padx=10, pady=5)
        
        # Load initialization metrics
        self.log_diagnostic_info()

    def log_diagnostic_info(self):
        self.update_log("AnonyVox Core Audio Module Initializing...")
        self.update_log(f"RVC Library Status: {'INSTALLED' if RVC_AVAILABLE else 'NOT INSTALLED'}")
        if RVC_AVAILABLE:
            self.update_log(f"GPU Hardware Context: {'CUDA:0 ACCELERATED' if self.cuda_active else 'CPU MODE (LATENCY ALERT)'}")
        else:
            self.update_log("RVC modulations bypassed. Local DSP operations remain active.")
            
        models = self.scan_models()
        self.update_log(f"Located {len(models)} RVC models (.pth) inside 'models/' folder.")

    def update_log(self, text):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, f"{timestamp}{text}\n")
        self.log_text.see(tk.END)

    def scan_models(self):
        if not os.path.exists(self.models_dir):
            return []
        return [f for f in os.listdir(self.models_dir) if f.endswith(".pth")]

    def refresh_model_list(self):
        models = self.scan_models()
        if not models:
            self.model_dropdown['values'] = ["No models found"]
            self.model_dropdown.set("No models found")
            self.model_dropdown.config(state="disabled")
            self.status_label.config(text="Warning: Drop some RVC models in models/ folder.", fg="#ef4444")
            self.rvc_enabled_var.set(False)
            self.rvc_chk.config(state="disabled")
        else:
            self.model_dropdown['values'] = models
            self.model_dropdown.set(models[0])
            self.model_dropdown.config(state="readonly")
            if RVC_AVAILABLE and self.rvc is not None:
                self.rvc_chk.config(state="normal")
            else:
                self.rvc_chk.config(state="disabled")
                self.rvc_enabled_var.set(False)
            # Automatically load first model in background thread
            self.on_model_selected()

    def get_model_index_pair(self, model_name):
        model_base = os.path.splitext(model_name)[0]
        index_files = [f for f in os.listdir(self.models_dir) if f.endswith(".index")]
        
        # Priority 1: Direct name matching
        exact_match = model_base + ".index"
        if exact_match in index_files:
            return os.path.join(self.models_dir, exact_match)
            
        # Priority 2: Substring or word token matches
        for f in index_files:
            # Check if one is a substring of the other
            if model_base.lower() in f.lower() or f.lower() in model_base.lower():
                return os.path.join(self.models_dir, f)
            
            # Check if they share any significant word tokens (>= 4 characters)
            tokens = [t.lower() for t in model_base.replace('_', ' ').replace('-', ' ').split() if len(t) >= 4]
            for token in tokens:
                if token in f.lower():
                    return os.path.join(self.models_dir, f)
                
        # Priority 3: Fallback to first index if single file
        if len(index_files) == 1:
            return os.path.join(self.models_dir, index_files[0])
            
        return None

    def on_model_selected(self, event=None):
        selected = self.model_var.get()
        if selected == "No models found" or not selected:
            return
            
        if not self.rvc:
            self.model_dropdown.config(state="readonly")
            self.status_label.config(text="⚠️ AI Modulator Unavailable (Dependencies missing)", fg="#ef4444")
            self.update_log("RVC Modulator bypassed: rvc-python library is not loaded.")
            self.rvc_enabled_var.set(False)
            self.rvc_chk.config(state="disabled")
            return

        model_path = os.path.join(self.models_dir, selected)
        index_path = self.get_model_index_pair(selected)
        
        self.model_dropdown.config(state="disabled")
        self.status_label.config(text=f"Loading target model: {selected}...", fg="#06b6d4")
        self.update_log(f"Model selection changed. Loading file: {selected}")
        
        def load_thread():
            try:
                with self.rvc_lock:
                    if self.rvc:
                        self.rvc.load_model(model_path, index_path=index_path or "")
                        self.current_model_path = model_path
                        self.current_index_path = index_path
                        
                # Update UI thread-safely
                self.root.after(0, lambda: self.on_load_success(selected, index_path))
            except Exception as e:
                self.root.after(0, lambda: self.on_load_failed(selected, str(e)))
                
        threading.Thread(target=load_thread, daemon=True).start()

    def on_load_success(self, model_name, index_path):
        self.model_dropdown.config(state="readonly")
        idx_name = os.path.basename(index_path) if index_path else "None"
        self.status_label.config(
            text=f"Ready • Loaded: {model_name} | GPU: {'CUDA:0' if self.cuda_active else 'CPU'}",
            fg="#06b6d4"
        )
        self.update_log(f"Mounted AI weights '{model_name}' successfully.")
        if index_path:
            self.update_log(f"Bound Index File: {idx_name}")
        else:
            self.update_log("Warning: Index mapping not found. Real-time similarity may leak.")

    def on_load_failed(self, model_name, err):
        self.model_dropdown.config(state="readonly")
        self.status_label.config(text=f"RVC Load Error: {model_name}", fg="#ef4444")
        self.update_log(f"Failed to load RVC Model weights {model_name}: {err}")
        messagebox.showerror("AnonyVox AI Engine Error", f"Failed to load model weights:\n{err}")

    def on_rvc_toggle(self):
        active = self.rvc_enabled_var.get()
        if active:
            if not self.current_model_path:
                self.update_log("Cannot engage AI pipeline: No weights loaded.")
                self.rvc_enabled_var.set(False)
            else:
                self.update_log("AI voice conversion pipeline ENGAGED.")
        else:
            self.update_log("AI voice conversion pipeline DISENGAGED. Standard DSP active.")

    def on_preset_selected(self, event=None):
        preset = self.preset_var.get()
        if preset == "Select Preset...":
            return
            
        self.update_log(f"Applying Preset Profile: {preset}")
        
        if preset == "Deep Male":
            self.pitch_val.set(-5)
            self.spectral_inversion_val.set(0.0)
            self.ring_mod_val.set(0)
            self.tremolo_depth_val.set(0.0)
            self.distortion_val.set(1.0)
        elif preset == "High Female":
            self.pitch_val.set(5)
            self.spectral_inversion_val.set(0.0)
            self.ring_mod_val.set(0)
            self.tremolo_depth_val.set(0.0)
            self.distortion_val.set(1.0)
        elif preset == "Astral Echo":
            self.pitch_val.set(0)
            self.spectral_inversion_val.set(0.40)
            self.ring_mod_val.set(150)
            self.tremolo_depth_val.set(0.20)
            self.distortion_val.set(1.0)
        elif preset == "Overdrive Cyborg":
            self.pitch_val.set(-3)
            self.spectral_inversion_val.set(0.0)
            self.ring_mod_val.set(400)
            self.tremolo_depth_val.set(0.0)
            self.distortion_val.set(4.0)
        elif preset == "Default Reset":
            self.pitch_val.set(0)
            self.spectral_inversion_val.set(0.0)
            self.ring_mod_val.set(0)
            self.tremolo_depth_val.set(0.0)
            self.distortion_val.set(1.0)
            
        self.status_label.config(text=f"Preset Loaded: {preset}", fg="#a78bfa")

    def auto_route_virtual_mic(self):
        found = False
        for friendly_name, idx in self.device_name_to_index.items():
            if "cable" in friendly_name.lower() or "virtual" in friendly_name.lower() or "vac" in friendly_name.lower():
                if friendly_name in self.output_dropdown['values']:
                    self.output_device_var.set(friendly_name)
                    found = True
                    self.update_log(f"Auto-routed output to Virtual Audio Cable: {friendly_name}")
                    self.on_device_changed()
                    
                    messagebox.showinfo(
                        "Virtual Mic Auto-Route Active",
                        f"Routed morphed audio output to: \n'{friendly_name}'\n\n"
                        "To make the OS and communication apps (Discord, Zoom, etc.) receive the morphed voice:\n"
                        "1. Open your Sound Control Panel or app Voice Settings.\n"
                        "2. Set the INPUT device (Microphone) to 'CABLE Output (VB-Audio Virtual Cable)'.\n"
                        "3. Your morphed voice is now routed directly as your virtual microphone!"
                    )
                    break
        if not found:
            self.update_log("Virtual Mic Auto-Route failed: No VB-Audio Cable detected.")
            messagebox.showwarning(
                "Virtual Cable Helper",
                "No Virtual Audio Cable was detected on your system.\n\n"
                "To route your morphed voice to Discord or games, you need to install a virtual audio driver.\n\n"
                "Recommended: Download free 'VB-CABLE Driver' from vb-audio.com, install it, and refresh hardware!"
            )

    def toggle_recording(self):
        if not self.engine_active:
            messagebox.showwarning("Recorder Error", "Please start the audio engine first.")
            return
            
        if not self.is_recording:
            self.is_recording = True
            self.recorded_blocks = []
            self.record_btn.config(text="⏹️ STOP RECORDING", bg="#ff2a5f", fg="#ffffff")
            self.update_log("Recording started... Morphing output is being captured.")
            self.status_label.config(text="Recording Active • Capturing Audio Output", fg="#ff2a5f")
        else:
            self.is_recording = False
            self.record_btn.config(text="🔴 START RECORDING", bg="#1c1a2e", fg="#ff2a5f")
            self.update_log("Recording stopped. Saving file...")
            
            if self.recorded_blocks:
                threading.Thread(target=self.save_recorded_file, daemon=True).start()
            else:
                self.update_log("Recording discarded: No audio captured.")

    def save_recorded_file(self):
        try:
            all_audio = np.concatenate(self.recorded_blocks)
            filename = f"anonyvox_recording_{int(time.time())}.wav"
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            
            pcm_data = (all_audio * 32767.0).astype(np.int16)
            with wave.open(filepath, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(44100)
                w.writeframes(pcm_data.tobytes())
                
            self.root.after(0, lambda: self.update_log(f"SUCCESS: Recording saved to {filename}"))
            self.root.after(0, lambda: self.status_label.config(text=f"Recording Saved: {filename}", fg="#00f0ff"))
        except Exception as e:
            self.root.after(0, lambda: self.update_log(f"Recording save failed: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Recorder Error", f"Failed to save recording:\n{e}"))

    def populate_devices(self):
        try:
            devices = sd.query_devices()
        except Exception as e:
            self.update_log(f"Error querying hardware audio devices: {e}")
            return
            
        input_list = []
        output_list = []
        self.device_name_to_index = {}
        
        for i, dev in enumerate(devices):
            name = dev['name']
            max_in = dev['max_input_channels']
            max_out = dev['max_output_channels']
            host_api_idx = dev['hostapi']
            
            try:
                api_name = sd.query_hostapis(host_api_idx)['name']
                friendly_name = f"{i}: {name} ({api_name})"
            except Exception:
                friendly_name = f"{i}: {name}"
                
            self.device_name_to_index[friendly_name] = i
            
            if max_in > 0:
                input_list.append(friendly_name)
            if max_out > 0:
                output_list.append(friendly_name)
                
        self.input_dropdown['values'] = input_list
        self.output_dropdown['values'] = output_list
        
        # Match OS defaults where possible
        try:
            default_in = sd.query_devices(kind='input')
            default_out = sd.query_devices(kind='output')
            
            # Map input
            for friendly in input_list:
                if self.device_name_to_index[friendly] == default_in['name'] or default_in['name'] in friendly:
                    self.input_dropdown.set(friendly)
                    break
            else:
                if input_list: self.input_dropdown.set(input_list[0])
                
            # Map output
            for friendly in output_list:
                if self.device_name_to_index[friendly] == default_out['name'] or default_out['name'] in friendly:
                    self.output_dropdown.set(friendly)
                    break
            else:
                if output_list: self.output_dropdown.set(output_list[0])
        except Exception:
            if input_list: self.input_dropdown.set(input_list[0])
            if output_list: self.output_dropdown.set(output_list[0])
            
        self.update_log(f"Audio Hardware Refreshed. Discovered {len(input_list)} inputs, {len(output_list)} outputs.")

    def on_scramble_start(self, event):
        self.scramble_active = True
        self.scramble_btn.config(bg="#ff2a5f", fg="#ffffff", text="SCRAMBLER DEPLOYED (BLUR ACTIVE)")
        self.status_label.config(text="AI Semantic Redaction Engaged • Scrambling PCM Frames", fg="#ff2a5f")
        self.update_log("ALERT: Scramble hotkey pressed. AI inference bypassed. Encrypted privacy carrier injected.")

    def on_scramble_stop(self, event):
        self.scramble_active = False
        self.scramble_btn.config(bg="#1c1a2e", fg="#ff2a5f", text="HOLD TO SCRAMBLE VOICE")
        self.status_label.config(text="Ready • Modulator Online", fg="#00f0ff")
        self.update_log("Scramble hotkey released. Duplex AI inference loop resumed.")

    def toggle_engine(self):
        if not self.engine_active:
            self.start_engine()
        else:
            self.stop_engine()

    def on_device_changed(self, event=None):
        if self.engine_active:
            self.update_log("Device changed. Restarting audio stream...")
            self.stop_engine()
            self.start_engine()

    def start_engine(self):
        in_device = self.input_device_var.get()
        out_device = self.output_device_var.get()
        
        in_idx = self.device_name_to_index.get(in_device)
        out_idx = self.device_name_to_index.get(out_device)
        
        if in_idx is None or out_idx is None:
            messagebox.showerror("Hardware Setup Error", "Invalid audio routing. Please check device connections.")
            return

        # Prepare processing queues
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
            
        self.accumulator = []
        self.carrier_phase_accumulator = 0
        self.worker_running = True
        
        # Spin up background audio computation thread
        self.worker_thread = threading.Thread(target=self.audio_worker_loop, daemon=True)
        self.worker_thread.start()
        
        try:
            # Separate Input and Output streams to prevent Windows host API mismatch crashes
            self.input_stream = sd.InputStream(
                device=in_idx,
                samplerate=44100,
                blocksize=self.stream_blocksize,
                channels=1,
                dtype='float32',
                callback=self.input_callback
            )
            
            self.output_stream = sd.OutputStream(
                device=out_idx,
                samplerate=44100,
                blocksize=self.stream_blocksize,
                channels=1,
                dtype='float32',
                callback=self.output_callback
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            self.engine_active = True
            self.start_btn.config(text="STOP ENGINE", bg="#ff2a5f", fg="#ffffff")
            self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#ff5c85"))
            self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg="#ff2a5f"))
            
            self.update_log("Duplex Input/Output streams ENGAGED.")
            self.status_label.config(text="Audio Modulator Active • Dual-Stream Routing Engaged", fg="#00f0ff")
        except Exception as e:
            self.update_log(f"Stream startup failure: {e}")
            messagebox.showerror("Engine Failure", f"Sounddevice failed to bind stream configuration:\n{e}")
            self.stop_engine()

    def stop_engine(self):
        self.engine_active = False
        self.worker_running = False
        
        # Stop input stream
        if hasattr(self, 'input_stream') and self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None
            
        # Stop output stream
        if hasattr(self, 'output_stream') and self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None
            
        # Clean up temp wave files securely
        for temp_file in [self.temp_in_path, self.temp_out_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        self.start_btn.config(text="START ENGINE", bg="#00f0ff", fg="#0b0a12")
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#80f8ff"))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg="#00f0ff"))
        
        self.update_log("Duplex Audio stream DISENGAGED.")
        self.status_label.config(text="System Stopped. Idle.", fg="#94a3b8")

    def input_callback(self, indata, frames, time_info, status):
        """Native sounddevice input thread callback."""
        if status:
            print(f"Input stream warning: {status}")

        input_mono = indata[:, 0].copy()
        
        # 1. AI Semantic Redaction: Bypasses model, outputs aggressively phase-inverted synthetic carrier
        if self.scramble_active:
            t = (self.carrier_phase_accumulator + np.arange(frames)) / 44100.0
            self.carrier_phase_accumulator = (self.carrier_phase_accumulator + frames) % 44100
            
            carrier = (np.sin(2 * np.pi * 280 * t) * np.cos(2 * np.pi * 730 * t) + 
                       np.sin(2 * np.pi * 1100 * t) + np.sin(2 * np.pi * 50 * t))
            
            scrambled = -input_mono * (1.5 + carrier)
            scrambled = np.clip(scrambled, -1.0, 1.0)
            
            self.output_queue.put(scrambled)
            return

        # 2. Parallel Processing Engine Active
        if self.engine_active:
            rvc_enabled = self.rvc_enabled_var.get()
            if rvc_enabled and self.rvc is not None and self.current_model_path:
                self.input_queue.put(input_mono)
            else:
                # Direct DSP processing path (low latency bypass)
                processed = self.apply_dsp_effects(input_mono)
                self.output_queue.put(processed)

    def output_callback(self, outdata, frames, time_info, status):
        """Native sounddevice output thread callback."""
        if status:
            print(f"Output stream warning: {status}")
            
        try:
            # Retrieve processed audio from queue
            processed_mono = self.output_queue.get_nowait()
            
            if self.monitor_audio_var.get():
                outdata[:, 0] = processed_mono
            else:
                outdata.fill(0) # Output silence to speaker but keep stream active
                
            # Record output blocks if active
            if hasattr(self, 'is_recording') and self.is_recording:
                self.recorded_blocks.append(processed_mono.copy())

            # Update visualizer with output audio
            with self.visualizer_lock:
                self.visualizer_data = processed_mono.copy()
        except queue.Empty:
            outdata.fill(0)

    def audio_worker_loop(self):
        """Background thread logic for RVC models and DSP engine runs."""
        while self.worker_running:
            try:
                # Wait for block data from sounddevice callback
                input_chunk = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rvc_enabled = self.rvc_enabled_var.get()

            # Case A: Bypass RVC, apply only parallel mathematical DSP matrix
            if not rvc_enabled or self.rvc is None or not self.current_model_path:
                processed_chunk = self.apply_dsp_effects(input_chunk)
                self.output_queue.put(processed_chunk)
                continue

            # Case B: AI RVC Modulator active
            self.accumulator.extend(input_chunk)
            
            # Accumulate enough voice data for RVC to run accurately (e.g. 0.23s buffer at 44100Hz = 10240 frames)
            target_buffer_size = 10240
            if len(self.accumulator) >= target_buffer_size:
                # Extract block
                voice_block = np.array(self.accumulator[:target_buffer_size], dtype=np.float32)
                
                # Keep 20% overlap to allow window blending
                overlap_size = int(target_buffer_size * 0.20)
                self.accumulator = self.accumulator[target_buffer_size - overlap_size:]

                try:
                    # Write float32 block to temporary PCM 16-bit WAV file
                    pcm_data = (voice_block * 32767.0).astype(np.int16)
                    with wave.open(self.temp_in_path, 'wb') as w_in:
                        w_in.setnchannels(1)
                        w_in.setsampwidth(2)
                        w_in.setframerate(44100)
                        w_in.writeframes(pcm_data.tobytes())

                    # Run RVC inference using CUDA GPU acceleration context
                    with self.rvc_lock:
                        # Synchronize parameters
                        self.rvc.set_params(
                            f0up_key=int(self.pitch_val.get()),
                            f0method="rmvpe",
                            index_rate=0.75,
                            protect=0.33
                        )
                        self.rvc.infer_file(
                            self.temp_in_path,
                            self.temp_out_path
                        )

                    # Read generated output wav from RVC
                    with wave.open(self.temp_out_path, 'rb') as w_out:
                        n_out_frames = w_out.getnframes()
                        out_rate = w_out.getframerate()
                        out_width = w_out.getsampwidth()
                        out_bytes = w_out.readframes(n_out_frames)

                    # Normalize bytes to float32
                    if out_width == 2:
                        out_samples = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    elif out_width == 4:
                        out_samples = np.frombuffer(out_bytes, dtype=np.float32)
                    else:
                        out_samples = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0

                    # Resample via interpolation if model uses standard 40000Hz or 32000Hz outputs
                    if out_rate != 44100 and len(out_samples) > 0:
                        orig_times = np.arange(len(out_samples)) / out_rate
                        new_times = np.arange(int(len(out_samples) * 44100 / out_rate)) / 44100.0
                        out_samples = np.interp(new_times, orig_times, out_samples).astype(np.float32)

                    # Apply post-RVC DSP effects (Inversion, Ring Mod, Tremolo, Distortion)
                    out_samples = self.apply_dsp_effects_post_rvc(out_samples)

                    # Slice processed audio into exact size of stream block size
                    num_samples_to_yield = target_buffer_size - overlap_size
                    if len(out_samples) >= num_samples_to_yield:
                        out_payload = out_samples[-num_samples_to_yield:]
                    else:
                        out_payload = out_samples

                    # Split output blocks and queue back to audio output
                    for pos in range(0, len(out_payload), self.stream_blocksize):
                        slice_block = out_payload[pos:pos+self.stream_blocksize]
                        if len(slice_block) < self.stream_blocksize:
                            # Pad final block
                            padding = np.zeros(self.stream_blocksize - len(slice_block), dtype=np.float32)
                            slice_block = np.concatenate([slice_block, padding])
                        self.output_queue.put(slice_block)

                except Exception as ex:
                    print(f"Engine worker error: {ex}")
                    # Fallback to bypass output
                    fallback_block = voice_block[-self.stream_blocksize:]
                    fallback_block = self.apply_dsp_effects(fallback_block)
                    self.output_queue.put(fallback_block)

    def pitch_shift_fft(self, audio_block, semitones):
        """Frequency-domain pitch shifter using spectral interpolation."""
        if semitones == 0 or len(audio_block) == 0:
            return audio_block
            
        scale = 2.0 ** (semitones / 12.0)
        
        # Perform Real FFT
        coeffs = np.fft.rfft(audio_block)
        num_bins = len(coeffs)
        
        # Map indices according to scale factor
        orig_indices = np.arange(num_bins)
        new_indices = orig_indices / scale
        
        # Interpolate real and imaginary parts separately
        real_interp = np.interp(orig_indices, new_indices, coeffs.real)
        imag_interp = np.interp(orig_indices, new_indices, coeffs.imag)
        
        new_coeffs = real_interp + 1j * imag_interp
        
        # Reconstruct waveform
        reconstructed = np.fft.irfft(new_coeffs, n=len(audio_block))
        return reconstructed.astype(np.float32)

    def apply_dsp_effects(self, audio_block):
        """Standard DSP processing pipeline for low-latency bypass / fallback."""
        processed = audio_block.copy()
        
        # 1. Pitch Shift DSP (Fallback)
        pitch_shift = self.pitch_val.get()
        if pitch_shift != 0:
            processed = self.pitch_shift_fft(processed, pitch_shift)
            
        # 2. Apply other effects
        return self.apply_dsp_effects_post_rvc(processed)

    def apply_dsp_effects_post_rvc(self, audio_block):
        """Applies non-pitch DSP modifiers (e.g. Inversion, Ring Mod, Tremolo, Dist)."""
        processed = audio_block.copy()
        
        # 1. Spectral Inversion
        si_factor = self.spectral_inversion_val.get()
        if si_factor > 0.0:
            processed = self.apply_spectral_inversion_dsp(processed, si_factor)
            
        # 2. Ring Modulation
        ring_freq = self.ring_mod_val.get()
        if ring_freq > 0:
            t = (self.carrier_phase_accumulator + np.arange(len(processed))) / 44100.0
            carrier = np.sin(2 * np.pi * ring_freq * t)
            processed = processed * carrier
            
        # 3. Tremolo
        trem_depth = self.tremolo_depth_val.get()
        if trem_depth > 0.0:
            t = (self.carrier_phase_accumulator + np.arange(len(processed))) / 44100.0
            lfo = 1.0 - trem_depth * (0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t))
            processed = processed * lfo
            
        # 4. Distortion
        dist_drive = self.distortion_val.get()
        if dist_drive > 1.0:
            processed = np.tanh(processed * dist_drive) / np.tanh(dist_drive)
            
        # Clip to prevent clipping noise
        processed = np.clip(processed, -1.0, 1.0)
        
        # Update phase accumulator
        self.carrier_phase_accumulator = (self.carrier_phase_accumulator + len(audio_block)) % 44100
        
        return processed.astype(np.float32)

    def apply_spectral_inversion_dsp(self, audio_block, inversion_factor):
        """Spectral Inversion (Frequency Mirroring) DSP matrix loop."""
        if len(audio_block) == 0 or inversion_factor <= 0.0:
            return audio_block

        # Perform Real FFT
        fft_coeffs = np.fft.rfft(audio_block)
        
        # Keep the DC component (index 0) intact to avoid massive DC offset
        dc_component = fft_coeffs[0]
        ac_components = fft_coeffs[1:]
        
        # Mirror the AC frequency bins (flip the spectrum upside down)
        # We conjugate the reversed coefficients to maintain correct phase relations
        mirrored_ac = np.conj(ac_components[::-1])
        
        # Blend the original spectrum and the mirrored spectrum based on the slider factor
        blended_ac = (1.0 - inversion_factor) * ac_components + inversion_factor * mirrored_ac
        
        # Reconstruct the full coefficients array
        new_coeffs = np.concatenate([[dc_component], blended_ac])
        
        # Execute Inverse Real FFT to return to the time domain wave
        reconstructed = np.fft.irfft(new_coeffs, n=len(audio_block))
        return reconstructed.astype(np.float32)

    def draw_visualizer(self):
        """Refreshes neon cyan audio waveform visualizer."""
        self.canvas.delete("wave")
        
        with self.visualizer_lock:
            data = self.visualizer_data.copy()
            
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Fallbacks for initial load size queries
        if width < 10: width = 450
        if height < 10: height = 150
        
        center_y = height / 2
        points = []
        
        # Map values to visualizer coordinate grid
        n_samples = len(data)
        if n_samples > 0:
            # Downsample to width coordinates
            for x in range(width):
                idx = int(x * n_samples / width)
                if idx < n_samples:
                    val = data[idx]
                    y = center_y - (val * center_y * 0.85)
                    points.append((x, y))
                    
        # Render dynamic waveform with bezier smooth curve lines and neon dual-glow
        if len(points) > 1:
            flat_points = [coordinate for pt in points for coordinate in pt]
            # Background glowing pink line
            self.canvas.create_line(
                flat_points, 
                fill="#ff007f", 
                width=4, 
                tags="wave", 
                smooth=True
            )
            # Foreground sharp cyan line
            self.canvas.create_line(
                flat_points, 
                fill="#00f0ff", 
                width=2, 
                tags="wave", 
                smooth=True
            )
        else:
            self.canvas.create_line(0, center_y, width, center_y, fill="#ff007f", width=2, tags="wave")
            self.canvas.create_line(0, center_y, width, center_y, fill="#00f0ff", width=1, tags="wave")
            
        # 30 ms refresh rate (~33 FPS)
        self.root.after(30, self.draw_visualizer)

    def __del__(self):
        # Stop stream if app is deleted/force closed
        self.stop_engine()


if __name__ == "__main__":
    # Run application
    root = tk.Tk()
    app = AnonyVoxApp(root)
    
    # Ensure threads are killed on close
    def on_closing():
        app.stop_engine()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
