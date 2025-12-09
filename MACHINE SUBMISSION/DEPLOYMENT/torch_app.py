import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QLabel, QLineEdit, QMainWindow, QVBoxLayout, 
    QWidget, QMenu, QPushButton, QTextEdit, QScrollArea
)

class LyricsWindow(QMainWindow):
    """Window to display lyrics"""
    def __init__(self, track_info, lyrics):
        super().__init__()
        track_name = track_info.get('track', 'Unknown Track')
        artist_name = track_info.get('artist', 'Unknown Artist')
        
        self.setWindowTitle(f"Lyrics: {track_name} - {artist_name}")
        self.setMinimumSize(600, 500)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Track info at the top
        info_label = QLabel(f"<h2>{track_name}</h2><h3>{artist_name}</h3>")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Lyrics display area with scroll
        lyrics_text = QTextEdit()
        lyrics_text.setReadOnly(True)
        lyrics_text.setPlainText(lyrics if lyrics else "No lyrics found")
        lyrics_text.setFontFamily("Courier")
        lyrics_text.setFontPointSize(11)
        
        # Add scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(lyrics_text)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)
        
        # Status label

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Information Retrieval")

        # Labels
        self.label = QLabel()
        self.a_label = QLabel("Artist Name")
        self.t_label = QLabel("Track Name")
        
        # LineEdit
        self.a_input = QLineEdit()
        self.t_input = QLineEdit()

        # Buttons
        self.submit_button = QPushButton("Search For Track")
        self.submit_button.clicked.connect(self.search_song)
        
        
        # Single result display (replaces QListWidget)
        self.result_label = QLabel("Search Result:")
        self.result_display = QLabel("No track found yet")
        self.result_display.setWordWrap(True)
        
        # Store the current track result
        self.current_track = None
        self.lyrics = None
        
        # Store reference to lyrics window
        self.lyrics_window = None
        
        self.setMinimumSize(400, 500)

        layout = QVBoxLayout()
        layout.addWidget(self.a_label)
        layout.addWidget(self.a_input)
        layout.addWidget(self.t_label)
        layout.addWidget(self.t_input)
        layout.addWidget(self.label)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def search_song(self):
        artist_name = self.a_input.text()
        track_name = self.t_input.text()

        if not artist_name or not track_name:
            self.label.setText("Please enter both artist and track name")
            return
        
        import search
        self.submit_button.setEnabled(False)
        self.submit_button.setText("Searching...")

        import torch_pred

        try:
            # Get single result with lyrics
            result = search.deezer_search(artist_name, track_name) 
            print("Result:", result)
            
            # Check if result contains both track info and lyrics
            if isinstance(result, tuple) and len(result) == 2:
                self.current_track, self.lyrics = result
            else:
                # Handle case where search returns only track info
                self.current_track = result
                self.lyrics = None
                print("No lyrics returned")
            
            # Display the single result
            self.display_result(self.current_track)
            
            # Open lyrics window if we have lyrics
            if self.lyrics:
                self.open_lyrics_window()
            
        except Exception as e:
            print(f"Error during search: {e}")
            self.label.setText(f"Error: {str(e)}")
            self.result_display.setText("Search failed. Please try again.")
        
        finally:
            # Re-enable button
            self.submit_button.setEnabled(True)
            self.submit_button.setText("Search For Track")

    def display_result(self, track_data):
        """Display single search result"""
        if isinstance(track_data, dict):
            # Format the display text
            display_text = f"<b>{track_data.get('track', 'Unknown Track')}</b><br>"
            display_text += f"Artist: {track_data.get('artist', 'Unknown Artist')}<br>"
            display_text += f"Album: {track_data.get('album', 'Unknown Album')}<br>"
            
            # if self.lyrics:
            #     lyrics_preview = self.lyrics[:100] + "..." if len(self.lyrics) > 100 else self.lyrics
            #     display_text += f"<br><i>Lyrics preview:</i><br>{lyrics_preview}"
            # else:
            #     display_text += "<br><i>No lyrics available</i>"
            
            self.result_display.setText(display_text)
            

            
        else:
            self.result_display.setText("No track found. Please try a different search.")
            self.label.setText("Search completed - no results found")

    def open_lyrics_window(self):
        """Open a new window to display lyrics"""
        if self.current_track and self.lyrics:
            # Close existing lyrics window if open
            if self.lyrics_window:
                self.lyrics_window.close()
            
            # Create and show new lyrics window
            self.lyrics_window = LyricsWindow(self.current_track, self.lyrics)
            self.lyrics_window.show()

    def analyze_track(self):
        """Analyze the currently displayed track"""
        if self.current_track and isinstance(self.current_track, dict):
            self.label.setText(f"Analyzing: {self.current_track.get('track')}...")
            
            # Import and run analysis
            import search
            search.analyze_selected_track(self.current_track)
        else:
            self.label.setText("No track to analyze. Please search first.")

    def contextMenuEvent(self, e):
        context = QMenu(self)
        context.addAction(QAction("test 1", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(e.globalPos())

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()