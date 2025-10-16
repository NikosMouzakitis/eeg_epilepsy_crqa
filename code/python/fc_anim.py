# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 08:19:56 2025

@author: User
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Force Qt5 backend to prevent separate plot windows
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QLineEdit, QPushButton, QFrame, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

def compute_stats(rqa_data, epileptic_indices):
    """
    Compute statistics (mean, std, min, max) for each feature over upper triangle
    of FC matrices in epileptic segments.
    """
    stats = []
    num_features = 16
    i, j = np.triu_indices(22, k=1)  # Upper triangle indices excluding diagonal
    
    for f in range(num_features):
        feature_matrices = rqa_data[..., f][epileptic_indices]
        values = feature_matrices[:, i, j]  # Shape: (num_epileptic, num_edges)
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        stats.append({
            'Feature': f + 1,
            'Mean': f"{mean_val:.4f}",
            'Std': f"{std_val:.4f}",
            'Min': f"{min_val:.4f}",
            'Max': f"{max_val:.4f}"
        })
    
    return pd.DataFrame(stats)

def plot_fc_topographic_interactive(rqa_data, initial_feature_idx=0, initial_threshold=0.30):
    """
    Visualizes functional connectivity as a topographic scalp plot for all epileptic segments in rqa_data
    interactively with PyQt5-based GUI (textboxes for feature index, threshold, Next/Previous buttons) 
    in a dedicated toolbar frame on top of the plot. Includes a stats table on the side.
    Nodes are placed at approximate midpoints of bipolar channels, 
    edges drawn based on FC strength. Fixed color range 0-1 for edge widths.
    
    Args:
        rqa_data: np.array, shape [num_windows, channels, channels, features+1 label]
        initial_feature_idx: int, initial feature index to start with (0-based)
        initial_threshold: float, initial threshold value
    """
    # Extract labels (last feature in each sample) - fixed
    labels = rqa_data[:, 0, 0, -1].astype(int)     # shape: [num_windows]
    
    # Get epileptic indices - fixed
    epileptic_indices = np.where(labels == 1)[0]
    
    if len(epileptic_indices) == 0:
        print("No epileptic segments found.")
        return
    
    num_features = 16  # Assuming 16 features based on context
    
    print(f"Interactive topographic visualization for {len(epileptic_indices)} epileptic segments.")
    print("Use textboxes to set feature index (0-15) and threshold, and buttons to navigate.")
    
    # Channel names
    channel_names = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
    ]
    
    # Approximate 2D positions (x, y) for each channel (midpoints of bipolar pairs)
    # Scaled to fit within -1 to 1
    positions = np.array([
        [-0.55, 0.65],   # 0: FP1-F7
        [-0.75, 0.225],  # 1: F7-T7
        [-0.75, -0.225], # 2: T7-P7
        [-0.55, -0.65],  # 3: P7-O1
        [-0.425, 0.75],  # 4: FP1-F3
        [-0.5, 0.45],    # 5: F3-C3
        [-0.5, 0],       # 6: C3-P3
        [-0.425, -0.55], # 7: P3-O1
        [0.425, 0.75],   # 8: FP2-F4
        [0.5, 0.45],     # 9: F4-C4
        [0.5, 0],        # 10: C4-P4
        [0.425, -0.55],  # 11: P4-O2
        [0.55, 0.65],    # 12: FP2-F8
        [0.75, 0.225],   # 13: F8-T8
        [0.75, -0.225],  # 14: T8-P8
        [0.55, -0.65],   # 15: P8-O2
        [0, 0.6],        # 16: FZ-CZ
        [0, 0],          # 17: CZ-PZ
        [-0.75, -0.225], # 18: P7-T7 (same as 2)
        [-0.7, 0.075],   # 19: T7-FT9
        [0, 0.15],       # 20: FT9-FT10
        [0.7, 0.075]     # 21: FT10-T8
    ])
    
    class TopoWindow(QMainWindow):
        def __init__(self, rqa_data, initial_feature_idx, initial_threshold, epileptic_indices, positions, channel_names, stats_df):
            super(TopoWindow, self).__init__()
            self.rqa_data = rqa_data
            self.epileptic_indices = epileptic_indices
            self.positions = positions
            self.channel_names = channel_names
            self.stats_df = stats_df
            self.current_index = 0
            self.current_feature_idx = initial_feature_idx
            self.current_threshold = initial_threshold
            
            self.setWindowTitle("Interactive Topographic FC Visualization")
            self.setGeometry(100, 100, 1400, 900)
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QLabel {
                    font-weight: bold;
                    color: #333333;
                }
                QLineEdit {
                    padding: 5px;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                }
                QPushButton {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    text-align: center;
                    font-weight: bold;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QTableWidget {
                    gridline-color: #dddddd;
                    alternate-background-color: #f9f9f9;
                }
                QHeaderView::section {
                    background-color: #e0e0e0;
                    padding: 4px;
                    border: 1px solid #cccccc;
                    font-weight: bold;
                }
            """)
            
            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            
            # Main layout
            main_layout = QVBoxLayout(central)
            main_layout.setSpacing(10)
            main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Toolbar
            toolbar = QFrame()
            toolbar.setFrameStyle(QFrame.Raised)
            toolbar_layout = QHBoxLayout(toolbar)
            toolbar_layout.setSpacing(10)
            
            # Feature Index
            feature_label = QLabel("Feature Index (0-15):")
            font = QFont()
            font.setPointSize(10)
            feature_label.setFont(font)
            toolbar_layout.addWidget(feature_label)
            self.feature_edit = QLineEdit(str(initial_feature_idx))
            self.feature_edit.setMaximumWidth(80)
            toolbar_layout.addWidget(self.feature_edit)
            
            # Threshold
            thresh_label = QLabel("Threshold:")
            thresh_label.setFont(font)
            toolbar_layout.addWidget(thresh_label)
            self.thresh_edit = QLineEdit(str(initial_threshold))
            self.thresh_edit.setMaximumWidth(80)
            toolbar_layout.addWidget(self.thresh_edit)
            
            # Navigation buttons
            self.prev_btn = QPushButton("Previous")
            self.prev_btn.clicked.connect(self.prev_slide)
            toolbar_layout.addWidget(self.prev_btn)
            
            self.next_btn = QPushButton("Next")
            self.next_btn.clicked.connect(self.next_slide)
            toolbar_layout.addWidget(self.next_btn)
            
            # Status label
            self.status_label = QLabel("")
            self.status_label.setFont(font)
            toolbar_layout.addStretch()
            toolbar_layout.addWidget(self.status_label)
            
            main_layout.addWidget(toolbar)
            
            # Splitter for plot and stats
            splitter = QSplitter(Qt.Horizontal)
            
            # Plot area
            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.canvas = FigureCanvas(self.fig)
            plot_layout.addWidget(self.canvas)
            splitter.addWidget(plot_widget)
            splitter.setStretchFactor(0, 2)  # Plot takes more space
            
            # Stats table
            stats_widget = QWidget()
            stats_layout = QVBoxLayout(stats_widget)
            stats_layout.setContentsMargins(0, 0, 0, 0)
            
            stats_label = QLabel("Statistics per Feature (Epileptic Segments - Upper Triangle)")
            stats_label.setFont(QFont("Arial", 10, QFont.Bold))
            stats_layout.addWidget(stats_label)
            
            self.table = QTableWidget()
            self.table.setRowCount(len(stats_df))
            self.table.setColumnCount(len(stats_df.columns))
            self.table.setHorizontalHeaderLabels(stats_df.columns)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.setAlternatingRowColors(True)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            
            # Populate table
            for row in range(len(stats_df)):
                for col in range(len(stats_df.columns)):
                    item = QTableWidgetItem(str(stats_df.iloc[row, col]))
                    self.table.setItem(row, col, item)
            
            stats_layout.addWidget(self.table)
            splitter.addWidget(stats_widget)
            splitter.setStretchFactor(1, 1)
            
            main_layout.addWidget(splitter)
            
            # Connect signals for updates
            self.feature_edit.returnPressed.connect(self.update_from_entry)
            self.feature_edit.editingFinished.connect(self.update_from_entry)
            self.thresh_edit.returnPressed.connect(self.update_from_entry)
            self.thresh_edit.editingFinished.connect(self.update_from_entry)
            
            # Initial plot
            self.update_plot()
        
        def update_plot(self):
            self.ax.clear()
            
            # Draw head outline (simple ellipse)
            head = Ellipse((0, 0), 2, 2, facecolor='lightgray', edgecolor='black', linewidth=2)
            self.ax.add_patch(head)
            self.ax.set_xlim(-1.2, 1.2)
            self.ax.set_ylim(-1.2, 1.2)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            
            # Extract current feature matrices
            feature_matrices = self.rqa_data[..., self.current_feature_idx]
            fc_matrix = feature_matrices[self.epileptic_indices[self.current_index]]
            
            # Apply threshold if given
            if self.current_threshold is not None:
                fc_matrix = np.where(fc_matrix >= self.current_threshold, fc_matrix, 0)
            
            # Normalize for line width (0-1 to 1-10 width)
            max_val = 1.0  # since fixed range
            widths = np.clip(fc_matrix / max_val * 10, 0, 10)
            
            # Draw edges (upper triangle to avoid duplicates)
            for i in range(22):
                for j in range(i+1, 22):
                    if fc_matrix[i, j] > 0:
                        self.ax.plot([self.positions[i, 0], self.positions[j, 0]], 
                                   [self.positions[i, 1], self.positions[j, 1]],
                                   color='red', linewidth=widths[i, j], alpha=0.6)
            
            # Plot nodes
            self.ax.scatter(self.positions[:, 0], self.positions[:, 1], s=100, color='blue', zorder=5)
            
            # Add labels (shortened for visibility)
            short_names = [name.replace('-', '\n') for name in self.channel_names]  # Stack labels
            for i, name in enumerate(short_names):
                self.ax.text(self.positions[i, 0], self.positions[i, 1], name, 
                           ha='center', va='center', fontsize=6)
            
            fname = f"Feature {self.current_feature_idx + 1}"
            thresh_str = f"{self.current_threshold}" if self.current_threshold is not None else "None"
            self.ax.set_title(f"Topographic FC - Epileptic Window {self.epileptic_indices[self.current_index]} ({fname})\n"
                            f"(Red lines: connections >= {thresh_str}, width ~ strength)",
                              fontsize=12, fontweight='bold')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Update status
            self.status_label.setText(f"Window: {self.current_index + 1}/{len(self.epileptic_indices)} | Feature: {self.current_feature_idx + 1}")
        
        def update_from_entry(self):
            updated = False
            
            # Update feature index
            try:
                new_feature = int(self.feature_edit.text())
                if 0 <= new_feature <= 15:
                    self.current_feature_idx = new_feature
                    updated = True
                else:
                    QMessageBox.warning(self, "Invalid Input", "Feature index must be between 0 and 15.")
                    self.feature_edit.setText(str(self.current_feature_idx))
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Feature index must be an integer.")
                self.feature_edit.setText(str(self.current_feature_idx))
            
            # Update threshold
            try:
                new_threshold = float(self.thresh_edit.text())
                if new_threshold >= 0:
                    self.current_threshold = new_threshold
                    updated = True
                else:
                    QMessageBox.warning(self, "Invalid Input", "Threshold must be >= 0.")
                    self.thresh_edit.setText(str(self.current_threshold))
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Threshold must be a number.")
                self.thresh_edit.setText(str(self.current_threshold))
            
            if updated:
                self.update_plot()
        
        def next_slide(self):
            self.current_index = (self.current_index + 1) % len(self.epileptic_indices)
            self.update_plot()
        
        def prev_slide(self):
            self.current_index = (self.current_index - 1) % len(self.epileptic_indices)
            self.update_plot()
    
    # Compute stats
    stats_df = compute_stats(rqa_data, epileptic_indices)
    
    # Create and run the application
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    window = TopoWindow(rqa_data, initial_feature_idx, initial_threshold, epileptic_indices, positions, channel_names, stats_df)
    window.show()
    
    app.exec_()

# Load only m1
m1 = np.load("p1_03.npy")

# Replace NaN and Inf values with 0
m1 = np.nan_to_num(m1, nan=0.0, posinf=0.0, neginf=0.0)

print("------------------------------_")
print("m1 shape")
print(m1.shape)

# Start the interactive topographic visualization
plot_fc_topographic_interactive(m1, initial_feature_idx=0, initial_threshold=0.30)