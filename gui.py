"""
Whisper Audio Transcription GUI
------------------------------
A graphical interface for OpenAI's Whisper speech recognition model.

Features:
- Multiple model size options (tiny to large)
- Local model caching
- Progress indication for downloads and processing
- Clean dark-themed output display
- Timestamp support
- Error handling and status updates

The GUI is built with tkinter and follows these design principles:
- Clear user feedback for all operations
- Thread-safe UI updates
- Consistent dark theme for output
- Proper state management during operations
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from transcribe import setup_whisper, transcribe_audio, MODELS
import threading
import os

class WhisperGUI:
    """
    Main GUI class for the Whisper transcription application.
    
    Handles:
    - Model selection and loading
    - Audio file selection
    - Transcription display
    - Progress indication
    - State management
    """

    def __init__(self, root):
        """Initialize the GUI with all its components."""
        self.root = root
        self.root.title("Whisper Transcription")
        self.root.geometry("900x700")
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use 'clam' theme for modern look
        
        # Configure styles
        self.style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), padding=10)
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 10, 'italic'))
        self.style.configure('Action.TButton', font=('Helvetica', 11), padding=10)
        self.style.configure('Dark.TFrame', background='#2b2b2b')
        
        # Main frame with padding and background
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(
            self.main_frame,
            text="Whisper Audio Transcription",
            style='Title.TLabel'
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model selection frame
        model_frame = ttk.LabelFrame(
            self.main_frame,
            text="Model Selection",
            padding="10"
        )
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Create a frame for the model selection and button row
        controls_frame = ttk.Frame(model_frame)
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        controls_frame.columnconfigure(1, weight=1)  # Make space between model and button flexible
        
        # Model label and combo on the left
        ttk.Label(
            controls_frame,
            text="Model:",
            style='Header.TLabel'
        ).grid(row=0, column=0, padx=(0, 10))
        
        self.model_var = tk.StringVar(value='small')
        self.model_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.model_var,
            values=list(MODELS.keys()),
            width=20,
            state='readonly'
        )
        self.model_combo.grid(row=0, column=1, sticky=tk.W)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # File selection button on the right
        self.select_button = ttk.Button(
            controls_frame,
            text="Select Audio File",
            command=self.select_file,
            style='Action.TButton',
            state='disabled'
        )
        self.select_button.grid(row=0, column=2, padx=(20, 0))
        
        # Progress bar directly under the model frame
        self.progress = ttk.Progressbar(
            self.main_frame,
            mode='indeterminate'
        )
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10), padx=20)
        
        # Output frame
        output_frame = ttk.LabelFrame(
            self.main_frame,
            text="Transcription Output",
            padding="10"
        )
        output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create a frame to hold the ScrolledText with proper background
        text_container = ttk.Frame(output_frame, style='Dark.TFrame')
        text_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        text_container.columnconfigure(0, weight=1)
        text_container.rowconfigure(0, weight=1)
        
        self.output = scrolledtext.ScrolledText(
            text_container,
            wrap=tk.WORD,
            height=20,
            width=80,
            font=('Courier', 11),  # Use Courier for better fixed-width rendering
            background='#2b2b2b',
            foreground='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#404040',
            selectforeground='#ffffff',
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0,
            spacing1=0,  # Reduce spacing to prevent rendering artifacts
            spacing2=0,
            spacing3=0
        )
        self.output.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Update tag configurations for cleaner rendering
        self.output.tag_configure('header', 
            font=('Courier', 12, 'bold'), 
            foreground='#88c0d0',
            background='#2b2b2b'
        )
        self.output.tag_configure('timestamp', 
            font=('Courier', 11),
            foreground='#a3be8c',
            background='#2b2b2b'
        )
        self.output.tag_configure('text', 
            font=('Courier', 11),
            foreground='#ffffff',
            background='#2b2b2b'
        )
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        self.status.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Initialize pipeline
        self.pipe = None
        self.current_model = 'small'
        self.status_var.set("Initializing Whisper model...")
        threading.Thread(target=self.initialize_pipeline).start()
        
        # Make text widget read-only by default
        self.output.config(state='disabled')
    
    def initialize_pipeline(self):
        """
        Initialize or switch the Whisper model.
        
        Handles:
        - Model downloading
        - UI state during download
        - Error handling
        - Thread-safe UI updates
        """
        try:
            # Disable controls during initialization
            self.root.after(0, lambda: [
                self.select_button.config(state='disabled'),
                self.model_combo.config(state='disabled'),
                self.status_var.set("Downloading model... This may take a few minutes..."),
                self.progress.stop()  # Ensure progress bar is stopped during download
            ])
            
            self.pipe = setup_whisper(self.current_model)
            
            # Enable controls when ready
            self.root.after(0, lambda: [
                self.select_button.config(state='normal'),
                self.model_combo.config(state='readonly'),
                self.status_var.set("Ready"),
                self.progress.stop()  # Ensure progress bar is stopped
            ])
        except Exception as e:
            self.root.after(0, lambda: [
                self.select_button.config(state='disabled'),
                self.model_combo.config(state='readonly'),
                self.status_var.set(f"Error initializing: {str(e)}"),
                self.progress.stop()  # Ensure progress bar is stopped
            ])
    
    def on_model_change(self, event):
        """
        Handle model selection changes.
        
        Manages:
        - UI state during model switching
        - Progress indication
        - Thread-safe model initialization
        """
        new_model = self.model_var.get()
        if new_model != self.current_model:
            self.current_model = new_model
            # Disable controls during model change
            self.select_button.config(state='disabled')
            self.model_combo.config(state='disabled')
            self.progress.stop()  # Stop progress during download
            self.status_var.set(f"Downloading {new_model} model... This may take a few minutes...")
            threading.Thread(target=self.initialize_pipeline).start()
    
    def select_file(self):
        """
        Handle audio file selection via dialog.
        
        Supports:
        - Multiple audio formats
        - Error checking
        - Immediate transcription start
        """
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.transcribe_file(file_path)
    
    def transcribe_file(self, file_path):
        """
        Manage the transcription process.
        
        Features:
        - Progress indication
        - Thread-safe UI updates
        - Error handling
        - Formatted output with timestamps
        - State management during processing
        """
        if not os.path.exists(file_path):
            self.status_var.set("Error: File not found")
            return
            
        def transcribe():
            try:
                # Disable button and start progress without clearing text yet
                self.root.after(0, lambda: [
                    self.select_button.config(state='disabled'),
                    self.status_var.set(f"Transcribing {os.path.basename(file_path)}..."),
                    self.progress.start(10)
                ])
                
                result = transcribe_audio(file_path, self.pipe)
                
                def update_output():
                    try:
                        self.output.config(state='normal')
                        self.output.delete(1.0, tk.END)
                        
                        if result:
                            # Add full transcription
                            self.output.insert(tk.END, "Full Transcription:\n", 'header')
                            self.output.insert(tk.END, "\n")  # Manual spacing
                            self.output.insert(tk.END, result["text"].strip(), 'text')
                            self.output.insert(tk.END, "\n\n\n")  # Manual spacing
                            
                            # Add timestamped chunks
                            self.output.insert(tk.END, "Timestamped Chunks:\n", 'header')
                            self.output.insert(tk.END, "\n")  # Manual spacing
                            
                            for chunk in result["chunks"]:
                                timestamp = chunk["timestamp"]
                                if isinstance(timestamp, tuple):
                                    start, end = timestamp
                                    time_str = f"[{start:.2f}s -> {end:.2f}s]"
                                else:
                                    time_str = f"[{timestamp:.2f}s]"
                                
                                self.output.insert(tk.END, time_str + " ", 'timestamp')
                                self.output.insert(tk.END, chunk['text'].strip(), 'text')
                                self.output.insert(tk.END, "\n\n")  # Manual double spacing between chunks
                    finally:
                        # Always make text widget read-only and update UI state
                        self.output.config(state='disabled')  # Make text widget read-only
                        self.progress.stop()
                        self.select_button.config(state='normal')
                        self.status_var.set("Transcription complete")
                
                self.root.after(0, update_output)
                
            except Exception as e:
                def handle_error():
                    self.progress.stop()
                    self.select_button.config(state='normal')
                    self.status_var.set(f"Error: {str(e)}")
                self.root.after(0, handle_error)
        
        threading.Thread(target=transcribe).start()

def main():
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 